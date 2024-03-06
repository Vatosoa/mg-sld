from django.conf import settings
from django.conf.urls.static import static

from django.contrib import admin
from django.urls import path
from .views import postag_vos_view

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", postag_vos_view, name="postag_vos"),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)