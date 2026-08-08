using System.Diagnostics;
using System.IO.Pipes;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text.Json;
using System.Text.Json.Serialization;
using HIDMaestro;

namespace AiPlayer.ControllerBridge;

internal static class Program
{
    private const string PipeName = "ai-player-controller";
    private const int DefaultWatchdogMilliseconds = 250;

    public static async Task<int> Main(string[] args)
    {
        string command = args.FirstOrDefault()?.ToLowerInvariant() ?? "serve";

        try
        {
            return command switch
            {
                "install" => Install(),
                "probe" => await ProbeAsync(),
                "serve" => await ServeAsync(
                    ParseWatchdog(args),
                    ParseClientSid(args)
                ),
                _ => Usage(command),
            };
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"Controller bridge failed: {error.Message}");
            return 1;
        }
    }

    private static int Install()
    {
        RequireAdministrator();
        using var context = CreateContext();
        context.InstallDriver();
        Console.WriteLine("HIDMaestro driver installed and ready.");
        return 0;
    }

    private static async Task<int> ProbeAsync()
    {
        RequireAdministrator();
        using var context = CreateContext();
        context.InstallDriver();
        using var controller = CreateXboxController(context);
        SubmitNeutral(controller);
        Console.WriteLine(
            $"Created neutral system controller: {controller.Profile.Name}"
        );
        await Task.Delay(1000);
        SubmitNeutral(controller);
        Console.WriteLine("HIDMaestro neutral-controller probe passed.");
        return 0;
    }

    private static async Task<int> ServeAsync(
        int watchdogMilliseconds,
        SecurityIdentifier clientSid
    )
    {
        RequireAdministrator();

        int recovered = HMOemNameOverride.RecoverOrphans();
        if (recovered > 0)
        {
            Console.WriteLine($"Recovered {recovered} stale name override(s).");
        }

        using var context = CreateContext();
        context.InstallDriver();
        using var controller = CreateXboxController(context);
        SubmitNeutral(controller);

        using var shutdown = new CancellationTokenSource();
        Console.CancelKeyPress += (_, eventArgs) =>
        {
            eventArgs.Cancel = true;
            shutdown.Cancel();
        };

        Console.WriteLine($"System controller ready: {controller.Profile.Name}");
        Console.WriteLine(
            $"Pipe: \\\\.\\pipe\\{PipeName}; watchdog: {watchdogMilliseconds} ms"
        );
        Console.WriteLine("Press Ctrl+C to stop and remove the controller.");

        try
        {
            while (!shutdown.IsCancellationRequested)
            {
                await ServeOneClientAsync(
                    controller,
                    watchdogMilliseconds,
                    clientSid,
                    shutdown
                );
            }
        }
        catch (OperationCanceledException) when (shutdown.IsCancellationRequested)
        {
            // Normal Ctrl+C or client-requested shutdown.
        }
        finally
        {
            SubmitNeutral(controller);
        }

        return 0;
    }

    private static async Task ServeOneClientAsync(
        HMController controller,
        int watchdogMilliseconds,
        SecurityIdentifier clientSid,
        CancellationTokenSource shutdown
    )
    {
        await using NamedPipeServerStream pipe = CreatePipe(clientSid);

        Console.WriteLine("Waiting for AI client...");
        await pipe.WaitForConnectionAsync(shutdown.Token);
        Console.WriteLine("AI client connected.");

        using var reader = new StreamReader(pipe, leaveOpen: true);
        using var writer = new StreamWriter(pipe, leaveOpen: true)
        {
            AutoFlush = true,
        };

        await writer.WriteLineAsync(
            JsonSerializer.Serialize(new
            {
                type = "ready",
                backend = "HIDMaestro",
                profile = controller.Profile.Id,
                watchdog_ms = watchdogMilliseconds,
            })
        );

        long lastStateAt = Stopwatch.GetTimestamp();
        bool outputIsNeutral = true;
        bool disconnect = false;
        Task<string?>? pendingRead = null;

        try
        {
            while (
                pipe.IsConnected &&
                !disconnect &&
                !shutdown.IsCancellationRequested
            )
            {
                pendingRead ??= reader.ReadLineAsync();
                Task delay = Task.Delay(20, shutdown.Token);
                Task completed = await Task.WhenAny(pendingRead, delay);

                if (completed == pendingRead)
                {
                    string? line = await pendingRead;
                    pendingRead = null;
                    if (line is null)
                    {
                        break;
                    }

                    try
                    {
                        using JsonDocument document = JsonDocument.Parse(line);
                        string type = document.RootElement
                            .GetProperty("type")
                            .GetString() ?? "";

                        switch (type)
                        {
                            case "state":
                                ControllerState state =
                                    JsonSerializer.Deserialize<ControllerState>(line)
                                    ?? throw new InvalidDataException(
                                        "State payload is empty."
                                    );
                                state.Validate();
                                SubmitState(controller, state);
                                outputIsNeutral = state.IsNeutral;
                                lastStateAt = Stopwatch.GetTimestamp();
                                break;

                            case "reset":
                                SubmitNeutral(controller);
                                outputIsNeutral = true;
                                lastStateAt = Stopwatch.GetTimestamp();
                                break;

                            case "disconnect":
                                disconnect = true;
                                break;

                            case "shutdown":
                                shutdown.Cancel();
                                break;

                            default:
                                throw new InvalidDataException(
                                    $"Unknown command type '{type}'."
                                );
                        }
                    }
                    catch (Exception error) when (
                        error is JsonException or
                        InvalidDataException or
                        ArgumentOutOfRangeException
                    )
                    {
                        SubmitNeutral(controller);
                        outputIsNeutral = true;
                        await writer.WriteLineAsync(
                            JsonSerializer.Serialize(new
                            {
                                type = "error",
                                message = error.Message,
                            })
                        );
                    }
                }

                if (
                    !outputIsNeutral &&
                    Stopwatch.GetElapsedTime(lastStateAt).TotalMilliseconds >=
                        watchdogMilliseconds
                )
                {
                    SubmitNeutral(controller);
                    outputIsNeutral = true;
                    Console.WriteLine("Watchdog released stale AI input.");
                }
            }
        }
        catch (IOException)
        {
            // The AI process exited without a clean disconnect.
        }
        finally
        {
            SubmitNeutral(controller);
            Console.WriteLine("AI client disconnected; controls released.");
        }
    }

    private static HMContext CreateContext()
    {
        var context = new HMContext();
        int count = context.LoadDefaultProfiles();
        if (count == 0)
        {
            context.Dispose();
            throw new InvalidOperationException(
                "HIDMaestro did not load any embedded controller profiles."
            );
        }
        return context;
    }

    private static HMController CreateXboxController(HMContext context)
    {
        HMProfile profile = context.GetProfile("xbox-360-wired")
            ?? throw new InvalidOperationException(
                "HIDMaestro profile 'xbox-360-wired' is missing."
            );
        return context.CreateController(profile);
    }

    private static void SubmitState(
        HMController controller,
        ControllerState source
    )
    {
        var state = new HMGamepadState
        {
            Axes = HMGamepadStateHelpers.StandardAxes(
                controller.Profile,
                Stick(source.LeftX),
                Stick(source.LeftY),
                Stick(source.RightX),
                Stick(source.RightY),
                source.LeftTrigger,
                source.RightTrigger
            ),
            Buttons = source.Buttons,
            Hat = source.Hat,
        };
        controller.SubmitState(in state);
    }

    private static void SubmitNeutral(HMController controller)
    {
        var state = new HMGamepadState();
        controller.SubmitState(in state);
    }

    private static float Stick(float value) => (value + 1.0f) / 2.0f;

    private static int ParseWatchdog(string[] args)
    {
        int index = Array.IndexOf(args, "--watchdog-ms");
        if (index < 0)
        {
            return DefaultWatchdogMilliseconds;
        }
        if (
            index + 1 >= args.Length ||
            !int.TryParse(args[index + 1], out int value) ||
            value is < 50 or > 5000
        )
        {
            throw new ArgumentException(
                "--watchdog-ms must be an integer from 50 through 5000."
            );
        }
        return value;
    }

    private static SecurityIdentifier ParseClientSid(string[] args)
    {
        int index = Array.IndexOf(args, "--client-sid");
        string sid = index >= 0 && index + 1 < args.Length
            ? args[index + 1]
            : WindowsIdentity.GetCurrent().User?.Value
                ?? throw new InvalidOperationException(
                    "Could not determine the controller client's Windows SID."
                );

        try
        {
            return new SecurityIdentifier(sid);
        }
        catch (ArgumentException error)
        {
            throw new ArgumentException(
                $"--client-sid is not a valid Windows SID: {sid}",
                error
            );
        }
    }

    private static NamedPipeServerStream CreatePipe(
        SecurityIdentifier clientSid
    )
    {
        WindowsIdentity serverIdentity = WindowsIdentity.GetCurrent();
        SecurityIdentifier serverSid = serverIdentity.User
            ?? throw new InvalidOperationException(
                "Could not determine the bridge process Windows SID."
            );

        var security = new PipeSecurity();
        security.SetAccessRuleProtection(isProtected: true, preserveInheritance: false);
        security.AddAccessRule(
            new PipeAccessRule(
                serverSid,
                PipeAccessRights.FullControl,
                AccessControlType.Allow
            )
        );
        security.AddAccessRule(
            new PipeAccessRule(
                clientSid,
                PipeAccessRights.ReadWrite,
                AccessControlType.Allow
            )
        );

        return NamedPipeServerStreamAcl.Create(
            PipeName,
            PipeDirection.InOut,
            1,
            PipeTransmissionMode.Byte,
            PipeOptions.Asynchronous,
            0,
            0,
            security,
            HandleInheritability.None,
            0
        );
    }

    private static void RequireAdministrator()
    {
        using WindowsIdentity identity = WindowsIdentity.GetCurrent();
        var principal = new WindowsPrincipal(identity);
        if (!principal.IsInRole(WindowsBuiltInRole.Administrator))
        {
            throw new UnauthorizedAccessException(
                "HIDMaestro requires an elevated process. Run start.ps1 " +
                "and approve the Windows administrator prompt."
            );
        }
    }

    private static int Usage(string command)
    {
        Console.Error.WriteLine($"Unknown command '{command}'.");
        Console.Error.WriteLine(
            "Usage: ai-player-controller-bridge [install|probe|serve] " +
            "[--watchdog-ms 250] [--client-sid S-1-...]"
        );
        return 2;
    }
}

internal sealed class ControllerState
{
    [JsonPropertyName("left_x")]
    public float LeftX { get; init; }

    [JsonPropertyName("left_y")]
    public float LeftY { get; init; }

    [JsonPropertyName("right_x")]
    public float RightX { get; init; }

    [JsonPropertyName("right_y")]
    public float RightY { get; init; }

    [JsonPropertyName("left_trigger")]
    public float LeftTrigger { get; init; }

    [JsonPropertyName("right_trigger")]
    public float RightTrigger { get; init; }

    [JsonPropertyName("south")]
    public bool South { get; init; }

    [JsonPropertyName("east")]
    public bool East { get; init; }

    [JsonPropertyName("west")]
    public bool West { get; init; }

    [JsonPropertyName("north")]
    public bool North { get; init; }

    [JsonPropertyName("left_bumper")]
    public bool LeftBumper { get; init; }

    [JsonPropertyName("right_bumper")]
    public bool RightBumper { get; init; }

    [JsonPropertyName("back")]
    public bool Back { get; init; }

    [JsonPropertyName("start")]
    public bool Start { get; init; }

    [JsonPropertyName("left_stick")]
    public bool LeftStick { get; init; }

    [JsonPropertyName("right_stick")]
    public bool RightStick { get; init; }

    [JsonPropertyName("dpad_up")]
    public bool DpadUp { get; init; }

    [JsonPropertyName("dpad_down")]
    public bool DpadDown { get; init; }

    [JsonPropertyName("dpad_left")]
    public bool DpadLeft { get; init; }

    [JsonPropertyName("dpad_right")]
    public bool DpadRight { get; init; }

    [JsonIgnore]
    public HMButton Buttons =>
        (South ? HMButton.A : HMButton.None) |
        (East ? HMButton.B : HMButton.None) |
        (West ? HMButton.X : HMButton.None) |
        (North ? HMButton.Y : HMButton.None) |
        (LeftBumper ? HMButton.LeftBumper : HMButton.None) |
        (RightBumper ? HMButton.RightBumper : HMButton.None) |
        (Back ? HMButton.Back : HMButton.None) |
        (Start ? HMButton.Start : HMButton.None) |
        (LeftStick ? HMButton.LeftStick : HMButton.None) |
        (RightStick ? HMButton.RightStick : HMButton.None);

    [JsonIgnore]
    public HMHat Hat
    {
        get
        {
            int vertical = (DpadDown ? 1 : 0) - (DpadUp ? 1 : 0);
            int horizontal = (DpadRight ? 1 : 0) - (DpadLeft ? 1 : 0);
            return (horizontal, vertical) switch
            {
                (0, -1) => HMHat.North,
                (1, -1) => HMHat.NorthEast,
                (1, 0) => HMHat.East,
                (1, 1) => HMHat.SouthEast,
                (0, 1) => HMHat.South,
                (-1, 1) => HMHat.SouthWest,
                (-1, 0) => HMHat.West,
                (-1, -1) => HMHat.NorthWest,
                _ => HMHat.None,
            };
        }
    }

    [JsonIgnore]
    public bool IsNeutral =>
        LeftX == 0 && LeftY == 0 &&
        RightX == 0 && RightY == 0 &&
        LeftTrigger == 0 && RightTrigger == 0 &&
        Buttons == HMButton.None && Hat == HMHat.None;

    public void Validate()
    {
        ValidateRange(LeftX, -1, 1, nameof(LeftX));
        ValidateRange(LeftY, -1, 1, nameof(LeftY));
        ValidateRange(RightX, -1, 1, nameof(RightX));
        ValidateRange(RightY, -1, 1, nameof(RightY));
        ValidateRange(LeftTrigger, 0, 1, nameof(LeftTrigger));
        ValidateRange(RightTrigger, 0, 1, nameof(RightTrigger));
    }

    private static void ValidateRange(
        float value,
        float minimum,
        float maximum,
        string name
    )
    {
        if (!float.IsFinite(value) || value < minimum || value > maximum)
        {
            throw new ArgumentOutOfRangeException(
                name,
                value,
                $"Value must be finite and in [{minimum}, {maximum}]."
            );
        }
    }
}
