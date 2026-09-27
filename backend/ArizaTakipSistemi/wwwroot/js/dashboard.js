document.addEventListener("DOMContentLoaded", function () {
    // --- Durum guncelleme (mevcut davranis) ---------------------------------
    document.querySelectorAll(".durum-select").forEach(function (select) {
        select.addEventListener("change", function () {
            var id = select.getAttribute("data-id");
            var durum = select.value;

            fetch("/Dashboard/DurumGuncelle", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: "id=" + encodeURIComponent(id) + "&durum=" + encodeURIComponent(durum)
            })
                .then(function (response) {
                    if (!response.ok) {
                        alert("Durum güncellenemedi. Lütfen tekrar deneyin.");
                    }
                })
                .catch(function () {
                    alert("Sunucuya bağlanılamadı.");
                });
        });
    });

    // --- Harita ---------------------------------------------------------------
    var haritaEl = document.getElementById("arizaHarita");
    var veriEl = document.getElementById("harita-verisi");
    if (!haritaEl || !veriEl || typeof L === "undefined") {
        return;
    }

    var noktalar = [];
    try {
        noktalar = JSON.parse(veriEl.textContent || "[]");
    } catch (e) {
        noktalar = [];
    }

    var baslangicMerkez = noktalar.length ? [noktalar[0].enlem, noktalar[0].boylam] : [41.0082, 28.9784];
    var harita = L.map(haritaEl, { scrollWheelZoom: false }).setView(baslangicMerkez, 14);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "&copy; OpenStreetMap katkıda bulunanlar",
        maxZoom: 19
    }).addTo(harita);

    if (noktalar.length) {
        var sinirlar = [];
        noktalar.forEach(function (nokta) {
            var isaret = L.circleMarker([nokta.enlem, nokta.boylam], {
                radius: 9,
                color: nokta.renk,
                weight: 2,
                fillColor: nokta.renk,
                fillOpacity: 0.55
            }).addTo(harita);

            var fotoHtml = nokta.foto
                ? '<img src="' + nokta.foto + '" style="width:100%;border-radius:8px;margin-top:6px;" />'
                : "";

            isaret.bindPopup(
                '<div class="popup-title">' + nokta.tur + '</div>' +
                '<div class="popup-sub">' + nokta.durum + ' · ' + nokta.zaman + '</div>' +
                fotoHtml
            );

            sinirlar.push([nokta.enlem, nokta.boylam]);
        });

        if (sinirlar.length > 1) {
            harita.fitBounds(sinirlar, { padding: [24, 24] });
        }
    }
});
