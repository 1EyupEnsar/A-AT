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
