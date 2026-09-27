using System.Security.Claims;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Mvc;

namespace ArizaTakipSistemi.Controllers;

/// <summary>Admin paneli icin sifreyle giris/cikis. appsettings.json'daki
/// AdminAyarlari:Sifre degeriyle karsilastirir, dogruysa cookie ile oturum acar.</summary>
public class AccountController : Controller
{
    private readonly IConfiguration _config;

    public AccountController(IConfiguration config)
    {
        _config = config;
    }

    [HttpGet]
    public IActionResult Login(string? returnUrl)
    {
        ViewBag.ReturnUrl = returnUrl;
        return View();
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Login(string sifre, string? returnUrl)
    {
        var dogruSifre = _config["AdminAyarlari:Sifre"];

        if (string.IsNullOrEmpty(sifre) || sifre != dogruSifre)
        {
            ViewBag.Hata = "Şifre yanlış.";
            ViewBag.ReturnUrl = returnUrl;
            return View();
        }

        var claims = new List<Claim> { new(ClaimTypes.Name, "Personel") };
        var identity = new ClaimsIdentity(claims, CookieAuthenticationDefaults.AuthenticationScheme);
        await HttpContext.SignInAsync(CookieAuthenticationDefaults.AuthenticationScheme, new ClaimsPrincipal(identity));

        if (!string.IsNullOrEmpty(returnUrl) && Url.IsLocalUrl(returnUrl))
        {
            return Redirect(returnUrl);
        }
        return RedirectToAction("Index", "Admin");
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Logout()
    {
        await HttpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
        return RedirectToAction("Index", "Dashboard");
    }
}
