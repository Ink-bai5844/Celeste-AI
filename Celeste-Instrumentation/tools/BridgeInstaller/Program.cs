using System.Security.Cryptography;
using System.Text.Json;
using Mono.Cecil;
using Mono.Cecil.Cil;

// This tool never launches or loads the game into the CLR. Cecil reads metadata
// and writes a separate managed assembly; the original remains untouched.
static class Program
{
    static int Main(string[] args)
    {
        try
        {
            if (args.Length == 2 && args[0] == "inspect")
            {
                using var module = ModuleDefinition.ReadModule(Path.GetFullPath(args[1]));
                Print(Inspect(module, args[1]));
                return 0;
            }
            if (args.Length != 4 || args[0] != "patch")
                throw new ArgumentException("Usage: BridgeInstaller inspect GAME/Celeste.exe | patch SOURCE/Celeste.exe BRIDGE/CelesteInstrumentation.dll OUTPUT/Celeste.exe");

            string source = Path.GetFullPath(args[1]);
            string bridge = Path.GetFullPath(args[2]);
            string output = Path.GetFullPath(args[3]);
            string sourceHash = Hash(source);
            if (Canonical(source) == Canonical(output))
                throw new InvalidOperationException("Refusing to modify the source game assembly in place. Choose an isolated output copy.");
            if (Canonical(output).Contains("/Steam/steamapps/", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("Refusing to write inside a Steam installation; choose a project-local game copy.");
            if (new FileInfo(output).LinkTarget != null)
                throw new InvalidOperationException("The output assembly must not be a symbolic link.");

            using var resolver = new DefaultAssemblyResolver();
            resolver.AddSearchDirectory(Path.GetDirectoryName(source)!);
            resolver.AddSearchDirectory(Path.GetDirectoryName(bridge)!);
            using var game = ModuleDefinition.ReadModule(source, new ReaderParameters { AssemblyResolver = resolver, InMemory = true });
            using var instrumentation = ModuleDefinition.ReadModule(bridge, new ReaderParameters { AssemblyResolver = resolver, InMemory = true });
            if (game.Assembly.Name.Name != "Celeste" || !game.AssemblyReferences.Any(r => r.Name == "FNA"))
                throw new InvalidOperationException("Expected the native FNA Celeste managed assembly.");
            if (game.Assembly.Name.HasPublicKey)
                throw new InvalidOperationException("Strong-named game assemblies require an explicit signing strategy.");
            if (!instrumentation.AssemblyReferences.Any(r => r.Name == "FNA") || instrumentation.AssemblyReferences.Any(r => r.Name.StartsWith("Microsoft.Xna.Framework")))
                throw new InvalidOperationException("The bridge must be built against FNA, not Windows XNA.");
            var entry = game.EntryPoint ?? throw new InvalidOperationException("Game has no managed entry point.");
            if (!entry.HasBody || entry.DeclaringType.FullName != "Celeste.Celeste")
                throw new InvalidOperationException("Unexpected game entry-point layout.");
            var execute = instrumentation.GetType("Instrumentation.Entry")?.Methods.SingleOrDefault(m => m.Name == "Execute" && m.IsStatic && m.Parameters.Count == 0)
                ?? throw new InvalidOperationException("Bridge has no supported Entry.Execute method.");
            var existing = entry.Body.Instructions.Where(i => i.Operand is MethodReference m && m.DeclaringType.FullName == "Instrumentation.Entry" && m.Name == "Execute").ToList();
            if (existing.Count > 1) throw new InvalidOperationException("Duplicate bridge entry calls found.");
            if (existing.Count == 0)
            {
                var first = entry.Body.Instructions.First();
                entry.Body.GetILProcessor().InsertBefore(first, Instruction.Create(OpCodes.Call, game.ImportReference(execute)));
            }

            Directory.CreateDirectory(Path.GetDirectoryName(output)!);
            string temporary = output + ".bridge-new-" + Guid.NewGuid().ToString("N");
            try
            {
                game.Write(temporary);
                using (var check = ModuleDefinition.ReadModule(temporary))
                {
                    if (check.EntryPoint.Body.Instructions.Count(i => i.Operand is MethodReference m && m.DeclaringType.FullName == "Instrumentation.Entry" && m.Name == "Execute") != 1)
                        throw new InvalidOperationException("Written output did not retain exactly one bridge entry call.");
                }
                File.Move(temporary, output, true);
            }
            finally { if (File.Exists(temporary)) File.Delete(temporary); }
            bool sourceUnchanged = Hash(source) == sourceHash;
            if (!sourceUnchanged) throw new IOException("The source assembly changed while the isolated copy was being patched.");
            Print(new { schema = "celeste-fna-bridge-install-v1", source, source_sha256 = sourceHash, bridge, bridge_sha256 = Hash(bridge), output, output_sha256 = Hash(output), injected_into = entry.FullName, source_unchanged = sourceUnchanged, gameplay_launched = false });
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    static object Inspect(ModuleDefinition module, string path)
    {
        var names = new[] { "Celeste.Celeste", "Celeste.Player", "Celeste.Level", "Celeste.ZipMover", "Celeste.CassetteBlock", "Celeste.CassetteBlockManager", "Monocle.MInput", "Monocle.Engine" };
        return new {
            path = Path.GetFullPath(path), sha256 = Hash(path), assembly = module.Assembly.FullName,
            runtime = module.RuntimeVersion, entry_point = module.EntryPoint?.FullName,
            framework = module.Assembly.CustomAttributes.FirstOrDefault(a => a.AttributeType.FullName == "System.Runtime.Versioning.TargetFrameworkAttribute")?.ConstructorArguments.FirstOrDefault().Value,
            references = module.AssemblyReferences.Select(r => r.FullName).ToArray(),
            zip_iterator_types = module.GetType("Celeste.ZipMover")?.NestedTypes.Select(t => new { name = t.FullName, fields = t.Fields.Select(f => f.FieldType.FullName + " " + f.Name).ToArray(), move_next = t.Methods.FirstOrDefault(m => m.Name == "MoveNext")?.Body.Instructions.Select(i => i.ToString()).ToArray() }).ToArray(),
            entry_instructions = module.EntryPoint?.Body.Instructions.Select(i => i.ToString()).ToArray(),
            game_constructors = module.GetType("Celeste.Celeste")?.Methods.Where(m => m.IsConstructor).Select(m => new { method = m.FullName, instructions = m.Body.Instructions.Select(i => i.ToString()).ToArray() }).ToArray(),
            types = names.Select(name => module.GetType(name)).Where(t => t != null).Select(t => new { name = t!.FullName, fields = t.Fields.Select(f => f.FieldType.FullName + " " + f.Name).ToArray(), methods = t.Methods.Select(m => m.FullName).ToArray() }).ToArray()
        };
    }
    static string Hash(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
    static void Print(object value) => Console.WriteLine(JsonSerializer.Serialize(value, new JsonSerializerOptions { WriteIndented = true }));
    static string Canonical(string path)
    {
        path = Path.GetFullPath(path);
        var components = path.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        string result = Path.GetPathRoot(path)!;
        foreach (string component in components)
        {
            result = Path.Combine(result, component);
            FileSystemInfo info = Directory.Exists(result) ? new DirectoryInfo(result) : new FileInfo(result);
            if (info.LinkTarget != null) result = info.ResolveLinkTarget(true)!.FullName;
        }
        return result;
    }
}
