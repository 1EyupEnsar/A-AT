using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using ArizaTakipSistemi.Data;
using ArizaTakipSistemi.Models;

namespace ArizaTakipSistemi.Controllers;

/// <summary>Web panelini (Dashboard) sunan MVC controller.</summary>
public class DashboardController : Controller
{
    private readonly ApplicationDbContext _context;

    public DashboardController(ApplicationDbContext context)
    {
        _context = context;
    }

    public async Task<IActionResult> Index(ArizaTuru? tur, ArizaDurumu? durum)
    {
        var query = _context.Arizalar.AsQueryable();

        if (tur.HasValue)
        {
            query = query.Where(a => a.ArizaTuru == tur);
        }

        if (durum.HasValue)
        {
            query = query.Where(a => a.Durum == durum);
        }

        var liste = await query.OrderByDescending(a => a.TespitZamani).ToListAsync();

        ViewBag.SeciliTur = tur;
        ViewBag.SeciliDurum = durum;
        ViewBag.ToplamSayi = await _context.Arizalar.CountAsync();
        ViewBag.BekliyorSayi = await _context.Arizalar.CountAsync(a => a.Durum == ArizaDurumu.Bekliyor);
        ViewBag.OnarimdaSayi = await _context.Arizalar.CountAsync(a => a.Durum == ArizaDurumu.Onarimda);
        ViewBag.CozulduSayi = await _context.Arizalar.CountAsync(a => a.Durum == ArizaDurumu.Cozuldu);

        return View(liste);
    }

    [HttpPost]
    public async Task<IActionResult> DurumGuncelle(int id, ArizaDurumu durum)
    {
        var ariza = await _context.Arizalar.FindAsync(id);
        if (ariza is null)
        {
            return NotFound();
        }

        ariza.Durum = durum;
        await _context.SaveChangesAsync();

        return Ok();
    }
}
