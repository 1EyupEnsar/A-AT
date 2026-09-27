using ArizaTakipSistemi.Models;

namespace ArizaTakipSistemi.Data;

/// <summary>Veritabani bossa panelin bos gorunmemesi icin ornek kayitlar ekler.</summary>
public static class DbSeeder
{
    public static void Seed(ApplicationDbContext context)
    {
        if (context.Arizalar.Any())
        {
            return;
        }

        var simdi = DateTime.UtcNow;

        context.Arizalar.AddRange(
            new Ariza
            {
                ArizaTuru = ArizaTuru.Cukur,
                TespitZamani = simdi.AddHours(-3),
                FotografYolu = string.Empty,
                GuvenSkoru = 0.91,
                Durum = ArizaDurumu.Bekliyor,
                Enlem = 41.0082,
                Boylam = 28.9784,
                Siddet = ArizaSiddeti.Buyuk
            },
            new Ariza
            {
                ArizaTuru = ArizaTuru.TasanCopKutusu,
                TespitZamani = simdi.AddHours(-8),
                FotografYolu = string.Empty,
                GuvenSkoru = 0.87,
                Durum = ArizaDurumu.Onarimda,
                Enlem = 41.0105,
                Boylam = 28.9812
            },
            new Ariza
            {
                ArizaTuru = ArizaTuru.HasarliTabela,
                TespitZamani = simdi.AddDays(-1),
                FotografYolu = string.Empty,
                GuvenSkoru = 0.79,
                Durum = ArizaDurumu.Cozuldu,
                Enlem = 41.0058,
                Boylam = 28.9750
            }
        );

        context.SaveChanges();
    }
}
