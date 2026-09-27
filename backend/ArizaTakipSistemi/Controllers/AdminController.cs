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
