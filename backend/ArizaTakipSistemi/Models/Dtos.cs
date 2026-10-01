using System.Text.Json.Serialization;

namespace ArizaTakipSistemi.Models;

/// <summary>Python tespit servisinin POST /api/ariza istegiyle gonderdigi govde.</summary>
public class ArizaCreateDto
{
    public ArizaTuru ArizaTuru { get; set; }
    public DateTime TespitZamani { get; set; }
    public string FotografYolu { get; set; } = string.Empty;
    public double? GuvenSkoru { get; set; }
    public double? Enlem { get; set; }
    public double? Boylam { get; set; }
    public ArizaSiddeti? Siddet { get; set; }
}

/// <summary>Dashboard veya API uzerinden durum guncelleme istegi.</summary>
public class ArizaDurumuUpdateDto
{
    public ArizaDurumu Durum { get; set; }
}

/// <summary>python-detector/web_video_test.py'nin stdout'a yazdigi tek satirlik
/// JSON ciktisinin karsiligi.</summary>
public class VideoTestSonucu
{
    [JsonPropertyName("hata")]
    public string? Hata { get; set; }

    [JsonPropertyName("kare_sayisi")]
    public int KareSayisi { get; set; }

    [JsonPropertyName("tespitler")]
    public Dictionary<string, int>? Tespitler { get; set; }
}
