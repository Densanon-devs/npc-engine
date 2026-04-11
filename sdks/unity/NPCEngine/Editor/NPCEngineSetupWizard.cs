using System;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace NPCEngine.Editor
{
    /// <summary>
    /// Setup wizard that downloads the NPC Engine server binary and model.
    /// Opens automatically on first import or via Window → NPC Engine → Setup Wizard.
    /// </summary>
    public class NPCEngineSetupWizard : EditorWindow
    {
        // ── Configuration ──────────────────────────────────────

        /// <summary>GitHub release URL pattern. Replace {version} and {platform}.</summary>
        private const string ReleaseUrlPattern =
            "https://github.com/Densanon-devs/npc-engine/releases/latest/download/NPCEngine-server-{0}-latest.zip";

        /// <summary>Direct model download URL (HuggingFace).</summary>
        private const string ModelUrl =
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf";

        private const string ModelFileName = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
        private const long ExpectedModelSize = 491_400_032; // ~469MB

        // ── State ──────────────────────────────────────────────

        private enum SetupState { NotStarted, Downloading, Extracting, Complete, Error }

        private SetupState _state = SetupState.NotStarted;
        private string _statusMessage = "";
        private float _progress;
        private bool _serverInstalled;
        private bool _modelInstalled;
        private string _installPath;

        // ── Menu entry ─────────────────────────────────────────

        [MenuItem("Window/NPC Engine/Setup Wizard")]
        public static void ShowWindow()
        {
            var window = GetWindow<NPCEngineSetupWizard>("NPC Engine Setup");
            window.minSize = new Vector2(480, 400);
            window.RefreshStatus();
        }

        /// <summary>
        /// Auto-open on first import if server is not installed.
        /// </summary>
        [InitializeOnLoadMethod]
        private static void CheckFirstRun()
        {
            if (!IsServerInstalled())
            {
                // Delay to avoid opening during compilation
                EditorApplication.delayCall += () =>
                {
                    if (!IsServerInstalled())
                        ShowWindow();
                };
            }
        }

        // ── Status checks ──────────────────────────────────────

        private static string GetInstallPath()
        {
            return Path.Combine(Application.streamingAssetsPath, "NPCEngine");
        }

        private static bool IsServerInstalled()
        {
            string dir = GetInstallPath();
            string exe = Path.Combine(dir, GetServerBinaryName());
            return File.Exists(exe);
        }

        private static bool IsModelInstalled()
        {
            string dir = GetInstallPath();
            string model = Path.Combine(dir, "models", ModelFileName);
            return File.Exists(model);
        }

        private static string GetServerBinaryName()
        {
#if UNITY_EDITOR_WIN
            return "npc-engine.exe";
#else
            return "npc-engine";
#endif
        }

        private static string GetPlatformName()
        {
#if UNITY_EDITOR_WIN
            return "windows";
#elif UNITY_EDITOR_OSX
            return "macos";
#else
            return "linux";
#endif
        }

        private void RefreshStatus()
        {
            _installPath = GetInstallPath();
            _serverInstalled = IsServerInstalled();
            _modelInstalled = IsModelInstalled();
        }

        // ── GUI ────────────────────────────────────────────────

        private void OnGUI()
        {
            GUILayout.Space(10);

            // Header
            var headerStyle = new GUIStyle(EditorStyles.boldLabel) { fontSize = 18 };
            GUILayout.Label("NPC Engine Setup", headerStyle);
            GUILayout.Space(5);
            GUILayout.Label(
                "Local AI-powered NPCs with memory, trust, gossip, and quests.\n" +
                "No cloud, no subscription, no internet required after setup.",
                EditorStyles.wordWrappedLabel);

            GUILayout.Space(15);
            DrawSeparator();
            GUILayout.Space(10);

            // Status
            GUILayout.Label("Installation Status", EditorStyles.boldLabel);
            GUILayout.Space(5);

            DrawStatusRow("Server binary", _serverInstalled);
            DrawStatusRow("AI model (Qwen2.5 0.5B, 469MB)", _modelInstalled);

            GUILayout.Space(15);

            // Install path
            EditorGUILayout.LabelField("Install path:", _installPath, EditorStyles.miniLabel);

            GUILayout.Space(10);
            DrawSeparator();
            GUILayout.Space(10);

            // Actions
            if (_state == SetupState.NotStarted || _state == SetupState.Error)
            {
                if (!_serverInstalled || !_modelInstalled)
                {
                    GUILayout.Label("Click below to download and install the NPC Engine.", EditorStyles.wordWrappedLabel);
                    GUILayout.Space(10);

                    if (!_serverInstalled)
                    {
                        if (GUILayout.Button("Download NPC Engine Server (~500MB)", GUILayout.Height(35)))
                        {
                            DownloadServer();
                        }
                        GUILayout.Space(5);
                    }

                    if (!_modelInstalled)
                    {
                        if (GUILayout.Button("Download AI Model (~469MB)", GUILayout.Height(35)))
                        {
                            DownloadModel();
                        }
                        GUILayout.Space(5);
                    }

                    if (!_serverInstalled && !_modelInstalled)
                    {
                        GUILayout.Space(5);
                        if (GUILayout.Button("Download Everything (~1GB)", GUILayout.Height(40)))
                        {
                            DownloadAll();
                        }
                    }
                }
                else
                {
                    GUILayout.Space(10);
                    var readyStyle = new GUIStyle(EditorStyles.boldLabel)
                    {
                        fontSize = 14,
                        normal = { textColor = new Color(0.2f, 0.8f, 0.2f) }
                    };
                    GUILayout.Label("Ready! Press Play to start talking to NPCs.", readyStyle);

                    GUILayout.Space(10);
                    GUILayout.Label(
                        "Quick start:\n" +
                        "1. Add NPCEngineServer and NPCEngineClient to a GameObject\n" +
                        "2. Call client.GenerateAsync(\"Hello!\", \"noah\")\n" +
                        "3. See Samples~ folder for a complete example",
                        EditorStyles.wordWrappedLabel);
                }
            }
            else if (_state == SetupState.Downloading || _state == SetupState.Extracting)
            {
                GUILayout.Label(_statusMessage, EditorStyles.wordWrappedLabel);
                EditorGUI.ProgressBar(
                    EditorGUILayout.GetControlRect(false, 20),
                    _progress, $"{(_progress * 100):F0}%");
            }
            else if (_state == SetupState.Complete)
            {
                var doneStyle = new GUIStyle(EditorStyles.boldLabel)
                {
                    normal = { textColor = new Color(0.2f, 0.8f, 0.2f) }
                };
                GUILayout.Label("Download complete!", doneStyle);
                GUILayout.Space(5);
                if (GUILayout.Button("Refresh Status"))
                {
                    RefreshStatus();
                    _state = SetupState.NotStarted;
                }
            }

            if (_state == SetupState.Error)
            {
                GUILayout.Space(10);
                var errorStyle = new GUIStyle(EditorStyles.wordWrappedLabel)
                {
                    normal = { textColor = Color.red }
                };
                GUILayout.Label(_statusMessage, errorStyle);
            }

            GUILayout.FlexibleSpace();
            DrawSeparator();
            GUILayout.Space(5);

            // Footer links
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Documentation", EditorStyles.linkLabel))
                Application.OpenURL("https://github.com/Densanon-devs/npc-engine");
            if (GUILayout.Button("API Reference", EditorStyles.linkLabel))
                Application.OpenURL("https://github.com/Densanon-devs/npc-engine/blob/master/GUIDE.md");
            GUILayout.EndHorizontal();
        }

        // ── Drawing helpers ────────────────────────────────────

        private void DrawStatusRow(string label, bool installed)
        {
            GUILayout.BeginHorizontal();
            var icon = installed ? "\u2705" : "\u274C"; // checkmark or X
            GUILayout.Label($"  {icon}  {label}", GUILayout.Width(400));
            GUILayout.EndHorizontal();
        }

        private void DrawSeparator()
        {
            var rect = EditorGUILayout.GetControlRect(false, 1);
            EditorGUI.DrawRect(rect, new Color(0.5f, 0.5f, 0.5f, 0.5f));
        }

        // ── Download logic ─────────────────────────────────────

        private async void DownloadServer()
        {
            _state = SetupState.Downloading;
            _statusMessage = "Downloading NPC Engine server...";
            _progress = 0;
            Repaint();

            try
            {
                string platform = GetPlatformName();
                string url = string.Format(ReleaseUrlPattern, platform);
                string zipPath = Path.Combine(Application.temporaryCachePath, "npc-engine-server.zip");

                await DownloadFile(url, zipPath);

                _state = SetupState.Extracting;
                _statusMessage = "Extracting server files...";
                _progress = 0.9f;
                Repaint();

                // Extract to StreamingAssets/NPCEngine/
                string installDir = GetInstallPath();
                Directory.CreateDirectory(installDir);
                ZipFile.ExtractToDirectory(zipPath, installDir, true);

                // Set executable permission on Unix
#if !UNITY_EDITOR_WIN
                string binary = Path.Combine(installDir, GetServerBinaryName());
                if (File.Exists(binary))
                {
                    var process = new System.Diagnostics.Process();
                    process.StartInfo.FileName = "chmod";
                    process.StartInfo.Arguments = $"+x \"{binary}\"";
                    process.Start();
                    process.WaitForExit();
                }
#endif

                File.Delete(zipPath);
                _state = SetupState.Complete;
                _statusMessage = "Server installed successfully!";
                RefreshStatus();
                AssetDatabase.Refresh();
            }
            catch (Exception ex)
            {
                _state = SetupState.Error;
                _statusMessage = $"Download failed: {ex.Message}\n\n" +
                                 "You can download manually from:\n" +
                                 "https://github.com/Densanon-devs/npc-engine/releases";
                Debug.LogError($"[NPCEngine] Server download failed: {ex}");
            }

            Repaint();
        }

        private async void DownloadModel()
        {
            _state = SetupState.Downloading;
            _statusMessage = "Downloading AI model (469MB)...";
            _progress = 0;
            Repaint();

            try
            {
                string installDir = GetInstallPath();
                string modelsDir = Path.Combine(installDir, "models");
                Directory.CreateDirectory(modelsDir);
                string destPath = Path.Combine(modelsDir, ModelFileName);

                await DownloadFile(ModelUrl, destPath);

                _state = SetupState.Complete;
                _statusMessage = "Model downloaded successfully!";
                RefreshStatus();
                AssetDatabase.Refresh();
            }
            catch (Exception ex)
            {
                _state = SetupState.Error;
                _statusMessage = $"Model download failed: {ex.Message}\n\n" +
                                 "You can download manually from:\n" +
                                 "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF";
                Debug.LogError($"[NPCEngine] Model download failed: {ex}");
            }

            Repaint();
        }

        private async void DownloadAll()
        {
            await Task.Run(() => { }); // yield to UI
            DownloadServer();
            // Model download will be triggered after server completes
            // via the "Download Model" button becoming visible
        }

        private async Task DownloadFile(string url, string destPath)
        {
            using var client = new HttpClient();
            client.Timeout = TimeSpan.FromMinutes(30);

            using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            long totalBytes = response.Content.Headers.ContentLength ?? -1;

            using var stream = await response.Content.ReadAsStreamAsync();
            using var fileStream = new FileStream(destPath, FileMode.Create, FileAccess.Write, FileShare.None);

            byte[] buffer = new byte[81920]; // 80KB chunks
            long bytesRead = 0;
            int read;

            while ((read = await stream.ReadAsync(buffer, 0, buffer.Length)) > 0)
            {
                await fileStream.WriteAsync(buffer, 0, read);
                bytesRead += read;

                if (totalBytes > 0)
                {
                    _progress = (float)bytesRead / totalBytes;
                    _statusMessage = $"Downloading... {bytesRead / (1024 * 1024)}MB / {totalBytes / (1024 * 1024)}MB";
                }
            }

            _progress = 1.0f;
        }
    }
}
