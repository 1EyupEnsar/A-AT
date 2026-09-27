using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using ArizaTakipSistemi.Data;
using ArizaTakipSistemi.Models;

namespace ArizaTakipSistemi.Controllers;

/// <summary>
/// Python tespit servisinin ve dashboard'un tukettigi REST API.
/// POST /api/ariza      -> Python tarafindan yeni tespit kaydi olusturur.
/// GET  /api/ariza       -> Turu/durumu filtrelenmis kayit listesi.
/// PUT  /api/ariza/{id}/durum -> Bir kaydin durumunu gunceller.
/// </summary>
[ApiController]
[Route("api/ariza")]
public class ArizaApiController : ControllerBase
{
    private readonly ApplicationDbContext _context;

    public ArizaApiController(ApplicationDbContext context)
    {
        _context = context;
    }

    [HttpPost]
    public async Task<IActionResult> Create([FromBody] ArizaCreateDto dto)
    {
        var ariza = new Ariza
        {
            ArizaTuru = dto.ArizaTuru,
            TespitZamani = dto.TespitZamani == default ? DateTime.UtcNow : dto.TespitZamani,
            FotografYolu = dto.FotografYolu,
            GuvenSkoru = dto.GuvenSkoru,
            Enlem = dto.Enlem,
            Boylam = dto.Boylam,
            Siddet = dto.Siddet,
            Durum = ArizaDurumu.Bekliyor
        };

        _context.Arizalar.Add(ariza);
        await _context.SaveChangesAsync();

        return CreatedAtAction(nameof(GetById), new { id = ariza.Id }, ariza);
    }

    [HttpGet]
    public async Task<IActionResult> GetAll([FromQuery] ArizaTuru? tur, [FromQuery] ArizaDurumu? durum)
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
        return Ok(liste);
    }

    [HttpGet("{id:int}")]
    public async Task<IActionResult> GetById(int id)
    {
        var ariza = await _context.Arizalar.FindAsync(id);
        return ariza is null ? NotFound() : Ok(ariza);
    }

    [HttpPut("{id:int}/durum")]
    public async Task<IActionResult> UpdateDurum(int id, [FromBody] ArizaDurumuUpdateDto dto)
    {
        var ariza = await _context.Arizalar.FindAsync(id);
        if (ariza is null)
        {
            return NotFound();
        }

        ariza.Durum = dto.Durum;
        await _context.SaveChangesAsync();

        return Ok(ariza);
    }
}
