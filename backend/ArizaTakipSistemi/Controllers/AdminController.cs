using System.Diagnostics;
using System.Text.Json;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using ArizaTakipSistemi.Data;
using ArizaTakipSistemi.Models;

namespace ArizaTakipSistemi.Controllers;

/// <summary>
/// Sadece sifreyle giris yapmis personelin erisebildigi yonetim paneli.
/// Buradan bir kaydin HER ALANI manuel olarak duzenlenebilir, elle yeni
/// kayit eklenebilir (kamyonetin kacirdigi bir arizayi personel bildirsin
/// diye) ya da hatali/gereksiz bir kayit tamamen silinebilir.
/// </summary>
[Authorize]
public class AdminController : Controller
{
    private readonly ApplicationDbContext _context;
    private readonly IWebHostEnvironment _env;

    public AdminController(ApplicationDbContext context, IWebHostEnvironment env)
    {
        _context = context;
        _env = env;
    }

    public async Task<IActionResult> Index(ArizaTuru? tur, ArizaDurumu? durum)
    {
        var query = _context.Arizalar.AsQueryable();

        if (tur.HasValue) query = query.Where(a => a.ArizaTuru == tur);
        if (durum.HasValue) query = query.Where(a => a.Durum == durum);

        var liste = await query.OrderByDescending(a => a.TespitZamani).ToListAsync();

        ViewBag.SeciliTur = tur;
        ViewBag.SeciliDurum = durum;
        return View(liste);
    }

    public async Task<IActionResult> Duzenle(int id)
    {
        var ariza = await _context.Arizalar.FindAsync(id);
        if (ariza is null) return NotFound();
        return View(ariza);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Duzenle(int id, Ariza form, IFormFile? fotografDosyasi)
    {
        var ariza = await _context.Arizalar.FindAsync(id);
        if (ariza is null) return NotFound();

        ariza.ArizaTuru = form.ArizaTuru;
        ariza.Durum = form.Durum;
        ariza.Siddet = form.Siddet;
        ariza.GuvenSkoru = form.GuvenSkoru;
        ariza.Enlem = form.Enlem;
        ariza.Boylam = form.Boylam;
        ariza.TespitZamani = form.TespitZamani;

        if (fotografDosyasi is { Length: > 0 })
        {
            ariza.FotografYolu = await FotografiKaydet(fotografDosyasi, ariza.ArizaTuru);
        }

        await _context.SaveChangesAsync();
        TempData["Mesaj"] = $"#{ariza.Id} numaralı kayıt güncellendi.";
        return RedirectToAction(nameof(Index));
    }

    public IActionResult Ekle()
    {
        return View(new Ariza { TespitZamani = DateTime.UtcNow, Durum = ArizaDurumu.Bekliyor });
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Ekle(Ariza form, IFormFile? fotografDosyasi)
    {
        var ariza = new Ariza
        {
            ArizaTuru = form.ArizaTuru,
            Durum = form.Durum,
            Siddet = form.Siddet,
            GuvenSkoru = form.GuvenSkoru,
            Enlem = form.Enlem,
            Boylam = form.Boylam,
            TespitZamani = form.TespitZamani == default ? DateTime.UtcNow : form.TespitZamani,
        };

        if (fotografDosyasi is { Length: > 0 })
        {
            ariza.FotografYolu = await FotografiKaydet(fotografDosyasi, ariza.ArizaTuru);
        }

        _context.Arizalar.Add(ariza);
        await _context.SaveChangesAsync();
        TempData["Mesaj"] = "Yeni kayıt eklendi (personel tarafından, manuel).";
        return RedirectToAction(nameof(Index));
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Sil(int id)
    {
        var ariza = await _context.Arizalar.FindAsync(id);
        if (ariza is null) return NotFound();

        if (!string.IsNullOrWhiteSpace(ariza.FotografYolu))
        {
            var tamYol = Path.Combine(_env.WebRootPath, ariza.FotografYolu.TrimStart('/').Replace('/', Path.DirectorySeparatorChar));
            if (System.IO.File.Exists(tamYol))
            {
                System.IO.File.Delete(tamYol);
            }
        }

        _context.Arizalar.Remove(ariza);
        await _context.SaveChangesAsync();
        TempData["Mesaj"] = $"#{id} numaralı kayıt silindi.";
        return RedirectToAction(nameof(Index));
    }

    public IActionResult VideoTesti()
    {
        return View();
    }

    /// <summary>
    /// Yuklenen videoyu python-detector/web_video_test.py'ye gonderir: script
    /// 3 uzman modeli video uzerinde calistirir, kutucuklu/etiketli sonucu
    /// tarayicida oynatilabilen bir mp4 olarak wwwroot/video-testleri altina
    /// yazar. Bu, main.py'nin aksine backend'e API uzerinden HICBIR KAYIT
    /// GONDERMEZ - salt gorsel bir "modelim bu videoda ne goruyor" testidir.
    /// </summary>
    [HttpPost]
    [ValidateAntiForgeryToken]
    [RequestSizeLimit(500_000_000)]
    public async Task<IActionResult> VideoTesti(IFormFile? videoDosyasi)
    {
        if (videoDosyasi is not { Length: > 0 })
        {
            ViewBag.Hata = "Lütfen bir video dosyası seç.";
            return View();
        }

        var kokDizin = Path.GetFullPath(Path.Combine(_env.ContentRootPath, "..", "..", "python-detector"));
        var pythonExe = Path.Combine(kokDizin, ".venv", "Scripts", "python.exe");
        var script = Path.Combine(kokDizin, "web_video_test.py");

        var girdiKlasoru = Path.Combine(kokDizin, "videos", "web_yuklenen");
        Directory.CreateDirectory(girdiKlasoru);
        var girdiYolu = Path.Combine(girdiKlasoru, $"{Guid.NewGuid():N}{Path.GetExtension(videoDosyasi.FileName)}");

        using (var stream = new FileStream(girdiYolu, FileMode.Create))
        {
            await videoDosyasi.CopyToAsync(stream);
        }

        var ciktiKlasoru = Path.Combine(_env.WebRootPath, "video-testleri");
        Directory.CreateDirectory(ciktiKlasoru);
        var ciktiAdi = $"{Guid.NewGuid():N}.mp4";
        var ciktiYolu = Path.Combine(ciktiKlasoru, ciktiAdi);

        var psi = new ProcessStartInfo
        {
            FileName = pythonExe,
            WorkingDirectory = kokDizin,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        psi.ArgumentList.Add(script);
        psi.ArgumentList.Add(girdiYolu);
        psi.ArgumentList.Add(ciktiYolu);

        string stdout, stderr;
        int cikisKodu;
        using (var process = Process.Start(psi)!)
        {
            stdout = await process.StandardOutput.ReadToEndAsync();
            stderr = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();
            cikisKodu = process.ExitCode;
        }

        System.IO.File.Delete(girdiYolu);

        var sonSatir = stdout.Trim().Split('\n').LastOrDefault(l => l.TrimStart().StartsWith("{"));
        VideoTestSonucu? sonuc = null;
        if (sonSatir != null)
        {
            try { sonuc = JsonSerializer.Deserialize<VideoTestSonucu>(sonSatir); }
            catch (JsonException) { /* asagida hata olarak ele alinacak */ }
        }

        if (sonuc?.Hata != null)
        {
            ViewBag.Hata = sonuc.Hata;
            return View();
        }
        if (sonuc is null || cikisKodu != 0)
        {
            ViewBag.Hata = "Video işlenemedi. Ayrıntı: " + (string.IsNullOrWhiteSpace(stderr) ? "(bilinmiyor)" : stderr[..Math.Min(500, stderr.Length)]);
            return View();
        }

        ViewBag.VideoUrl = $"/video-testleri/{ciktiAdi}";
        ViewBag.Tespitler = sonuc.Tespitler;
        ViewBag.KareSayisi = sonuc.KareSayisi;
        return View();
    }

    private async Task<string> FotografiKaydet(IFormFile dosya, ArizaTuru tur)
    {
        var uzanti = Path.GetExtension(dosya.FileName);
        var dosyaAdi = $"{tur}_{Guid.NewGuid().ToString("N")[..10]}{uzanti}";
        var klasor = Path.Combine(_env.WebRootPath, "uploads");
        Directory.CreateDirectory(klasor);
        var tamYol = Path.Combine(klasor, dosyaAdi);

        using (var stream = new FileStream(tamYol, FileMode.Create))
        {
            await dosya.CopyToAsync(stream);
        }

        return $"/uploads/{dosyaAdi}";
    }
}
