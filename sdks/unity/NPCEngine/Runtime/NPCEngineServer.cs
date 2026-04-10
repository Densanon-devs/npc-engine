using System;
using System.Diagnostics;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Events;
using Debug = UnityEngine.Debug;

namespace NPCEngine
{
    /// <summary>
    /// Manages the NPC Engine server binary as a child process.
    /// Attach to a GameObject to automatically start and stop the server
    /// alongside your Unity play session.
    /// </summary>
    public class NPCEngineServer : MonoBehaviour
    {
        [SerializeField]
        [Tooltip("Path to the NPC Engine server executable. Auto-detected if left empty.")]
        private string serverBinaryPath = "";

        [SerializeField]
        [Tooltip("Port the server listens on.")]
        private int port = 8000;

        [SerializeField]
        [Tooltip("Automatically start the server when entering Play mode.")]
        private bool autoStartOnPlay = true;

        [SerializeField]
        [Tooltip("Maximum seconds to wait for the server to become healthy.")]
        private float startupTimeoutSeconds = 30f;

        /// <summary>Fired when the server is healthy and ready to accept requests.</summary>
        public UnityEvent OnServerReady;

        /// <summary>Fired when the server encounters an error. Passes the error message.</summary>
        public UnityEvent<string> OnServerError;

        private Process _serverProcess;

        /// <summary>
        /// Returns true if the server process is currently running.
        /// </summary>
        public bool IsRunning
        {
            get
            {
                if (_serverProcess == null) return false;
                try { return !_serverProcess.HasExited; }
                catch { return false; }
            }
        }

        /// <summary>
        /// The port the server is configured to listen on.
        /// </summary>
        public int Port => port;

        private void Awake()
        {
            if (autoStartOnPlay)
            {
                StartServer();
            }
        }

        private void OnApplicationQuit()
        {
            StopServer();
        }

        private void OnDestroy()
        {
            StopServer();
        }

        /// <summary>
        /// Launches the NPC Engine server binary as a subprocess and polls
        /// the health endpoint until the server is ready.
        /// </summary>
        public async void StartServer()
        {
            if (IsRunning)
            {
                Debug.Log("[NPCEngine] Server is already running.");
                return;
            }

            string binaryPath = ResolveBinaryPath();
            if (string.IsNullOrEmpty(binaryPath))
            {
                string error = "Could not locate the NPC Engine server binary. Set serverBinaryPath in the Inspector.";
                Debug.LogError($"[NPCEngine] {error}");
                OnServerError?.Invoke(error);
                return;
            }

            Debug.Log($"[NPCEngine] Starting server: {binaryPath} --port {port}");

            try
            {
                var startInfo = new ProcessStartInfo
                {
                    FileName = binaryPath,
                    Arguments = $"--port {port}",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                _serverProcess = Process.Start(startInfo);

                if (_serverProcess == null || _serverProcess.HasExited)
                {
                    string error = "Failed to start the server process.";
                    Debug.LogError($"[NPCEngine] {error}");
                    OnServerError?.Invoke(error);
                    return;
                }

                // Capture output asynchronously so it doesn't block
                _serverProcess.BeginOutputReadLine();
                _serverProcess.BeginErrorReadLine();

                _serverProcess.OutputDataReceived += (sender, args) =>
                {
                    if (!string.IsNullOrEmpty(args.Data))
                        Debug.Log($"[NPCEngine Server] {args.Data}");
                };

                _serverProcess.ErrorDataReceived += (sender, args) =>
                {
                    if (!string.IsNullOrEmpty(args.Data))
                        Debug.LogWarning($"[NPCEngine Server] {args.Data}");
                };

                await PollHealthUntilReady();
            }
            catch (Exception ex)
            {
                string error = $"Exception starting server: {ex.Message}";
                Debug.LogError($"[NPCEngine] {error}");
                OnServerError?.Invoke(error);
            }
        }

        /// <summary>
        /// Stops the server process if it is running.
        /// </summary>
        public void StopServer()
        {
            if (!IsRunning) return;

            Debug.Log("[NPCEngine] Stopping server...");

            try
            {
                _serverProcess.Kill();
                _serverProcess.WaitForExit(5000);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[NPCEngine] Error stopping server: {ex.Message}");
            }
            finally
            {
                _serverProcess?.Dispose();
                _serverProcess = null;
            }
        }

        /// <summary>
        /// Polls the health endpoint until the server responds or the timeout is reached.
        /// </summary>
        private async Task PollHealthUntilReady()
        {
            float elapsed = 0f;
            float pollInterval = 0.5f;
            string healthUrl = $"http://127.0.0.1:{port}/health";

            while (elapsed < startupTimeoutSeconds)
            {
                if (!IsRunning)
                {
                    string error = "Server process exited unexpectedly during startup.";
                    Debug.LogError($"[NPCEngine] {error}");
                    OnServerError?.Invoke(error);
                    return;
                }

                try
                {
                    using (var request = UnityEngine.Networking.UnityWebRequest.Get(healthUrl))
                    {
                        var op = request.SendWebRequest();
                        while (!op.isDone)
                            await Task.Yield();

                        if (request.result == UnityEngine.Networking.UnityWebRequest.Result.Success)
                        {
                            Debug.Log("[NPCEngine] Server is ready.");
                            OnServerReady?.Invoke();
                            return;
                        }
                    }
                }
                catch
                {
                    // Server not ready yet; keep polling.
                }

                await Task.Delay((int)(pollInterval * 1000));
                elapsed += pollInterval;
            }

            string timeoutError = $"Server did not become healthy within {startupTimeoutSeconds}s.";
            Debug.LogError($"[NPCEngine] {timeoutError}");
            OnServerError?.Invoke(timeoutError);
        }

        /// <summary>
        /// Resolves the server binary path, checking the configured path first,
        /// then falling back to common locations relative to the project.
        /// </summary>
        private string ResolveBinaryPath()
        {
            // Use explicit path if set
            if (!string.IsNullOrEmpty(serverBinaryPath))
            {
                if (System.IO.File.Exists(serverBinaryPath))
                    return serverBinaryPath;

                Debug.LogWarning($"[NPCEngine] Configured binary path not found: {serverBinaryPath}");
            }

            // Auto-detect common locations
            string[] candidates = new[]
            {
                System.IO.Path.Combine(Application.streamingAssetsPath, "NPCEngine", "npc-engine"),
                System.IO.Path.Combine(Application.streamingAssetsPath, "NPCEngine", "npc-engine.exe"),
                System.IO.Path.Combine(Application.dataPath, "..", "npc-engine", "npc-engine"),
                System.IO.Path.Combine(Application.dataPath, "..", "npc-engine", "npc-engine.exe"),
            };

            foreach (string path in candidates)
            {
                if (System.IO.File.Exists(path))
                {
                    Debug.Log($"[NPCEngine] Auto-detected server binary: {path}");
                    return path;
                }
            }

            return null;
        }
    }
}
