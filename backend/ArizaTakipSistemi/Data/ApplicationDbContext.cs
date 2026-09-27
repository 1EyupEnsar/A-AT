using Microsoft.EntityFrameworkCore;
using ArizaTakipSistemi.Models;

namespace ArizaTakipSistemi.Data;

public class ApplicationDbContext : DbContext
{
    public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
        : base(options)
    {
    }

    public DbSet<Ariza> Arizalar => Set<Ariza>();
}
