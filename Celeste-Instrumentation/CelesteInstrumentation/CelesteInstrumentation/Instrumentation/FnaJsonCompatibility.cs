#if CELESTE_FNA
using System;
using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

// The game's bundled Mono distribution has no ASP.NET assemblies. Preserve the
// small serializer surface used by the bridge without System.Web.Extensions.
namespace System.Web.Script.Serialization
{
    internal sealed class JavaScriptSerializer
    {
        public string Serialize(object value)
        {
            return JsonConvert.SerializeObject(value, Formatting.None);
        }

        public object DeserializeObject(string json)
        {
            return ConvertToken(JToken.Parse(json));
        }

        private static object ConvertToken(JToken token)
        {
            JObject dictionary = token as JObject;
            if (dictionary != null)
            {
                var result = new Dictionary<string, object>(StringComparer.Ordinal);
                foreach (JProperty property in dictionary.Properties()) result[property.Name] = ConvertToken(property.Value);
                return result;
            }
            JArray array = token as JArray;
            if (array != null)
            {
                object[] result = new object[array.Count];
                for (int index = 0; index < array.Count; index++) result[index] = ConvertToken(array[index]);
                return result;
            }
            return ((JValue)token).Value;
        }
    }
}
#endif
