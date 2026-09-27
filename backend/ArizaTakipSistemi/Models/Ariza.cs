namespace ArizaTakipSistemi.Models;

/// <summary>Tespit edilen altyapi arizasinin turu. YOLOv8 tarafindan bulunan 3 sinifla eslesir.</summary>
public enum ArizaTuru
{
    Cukur,
    TasanCopKutusu,
    HasarliTabela
}

/// <summary>Bir arizanin islem sureci boyunca alabilecegi durumlar.</summary>
public enum ArizaDurumu
{
    Bekliyor,
    Onarimda,
    Cozuldu
}

/// <summary>
/// Tespit kutusunun kare alanina oraniyla hesaplanan kaba siddet tahmini.
/// Su an sadece Cukur icin anlamli (bkz. python-detector/config.py); diger
/// turlerde bos (null) birakilir.
/// </summary>
public enum ArizaSiddeti
{
    Kucuk,
    Orta,
    Buyuk
}

/// <summary>Veritabaninda saklanan tek bir ariza kaydi.</summary>
public class Ariza
{
    public int Id { get; set; }

    public ArizaTuru ArizaTuru { get; set; }

    public DateTime TespitZamani { get; set; }

    /// <summary>wwwroot altindaki goreli yol, orn: "/uploads/Cukur_ab12cd.jpg"</summary>
    public string FotografYolu { get; set; } = string.Empty;

    /// <summary>YOLOv8 modelinin tespit icin verdigi guven skoru (0-1 arasi).</summary>
    public double? GuvenSkoru { get; set; }

    public ArizaDurumu Durum { get; set; } = ArizaDurumu.Bekliyor;

    /// <summary>Ileride harita entegrasyonu icin ayrilmis, opsiyonel alanlar.</summary>
    public double? Enlem { get; set; }
    public double? Boylam { get; set; }

    /// <summary>Sadece Cukur turu icin doldurulur; digerlerinde null kalir.</summary>
    public ArizaSiddeti? Siddet { get; set; }
}
