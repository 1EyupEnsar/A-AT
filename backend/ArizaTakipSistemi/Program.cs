using System.Globalization;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.EntityFrameworkCore;
using ArizaTakipSistemi.Data;

// HTML <input type="number"> her zaman ondalik ayrac olarak "." kullanir
// (kultur bagimsiz). Sunucunun isletim sistemi kulturu tr-TR gibi "."yi
// binlik ayraci sayan bir kulturse, "41.02" gibi bir deger model binding
// sirasinda 4102'ye donusur. Bunu onlemek icin tum uygulamayi InvariantCulture
// ile calistiriyoruz (sayisal binding/formatlama her zaman "." kullanir).
CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllersWithViews()
    .AddJsonOptions(options =>
    {
        // Python'dan gelen JSON'da ArizaTuru/ArizaDurumu string olarak
        // gonderilir (orn. "Cukur"); enum'lari sayi degil metinle esleriz.
        options.JsonSerializerOptions.Converters.Add(new JsonStringEnumConverter());
    });

builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));

// Admin paneli sadece sifre ile korunuyor: girisli olmayan kullanicilar
// /Admin altina gitmeye calisinca /Account/Login'e yonlendirilir.
builder.Services.AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
    .AddCookie(options =>
    {
        options.LoginPath = "/Account/Login";
        options.AccessDeniedPath = "/Account/Login";
        options.ExpireTimeSpan = TimeSpan.FromHours(12);
        options.SlidingExpiration = true;
        options.Cookie.Name = "AsatAdminOturum";
    });
builder.Services.AddAuthorization();

var app = builder.Build();

using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
    db.Database.EnsureCreated();
    DbSeeder.Seed(db);
}

if (!app.Environment.IsDevelopment())
{
    // Not: Projede ayri bir Home/Error sayfasi yok (kucuk bir demo projesi
    // oldugu icin gereksiz gorulup eklenmedi); prod modda beklenmeyen bir
    // hata olursa kullaniciyi var olan bir sayfaya (Dashboard) yonlendiriyoruz.
    app.UseExceptionHandler("/Dashboard/Index");
}

app.UseStaticFiles();
app.UseRouting();

app.UseAuthentication();
app.UseAuthorization();

app.MapControllers();
app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Dashboard}/{action=Index}/{id?}");

app.Run();
