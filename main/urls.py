from django.urls import path
from . import views

urlpatterns = [
    # Trang chính
    path('', views.home, name='home'),
    path('index.html', views.home, name='index'),

    # Các trang thông tin
    path('about.html', views.about_view, name='about'),
    path('contact.html', views.contact_view, name='contact'),
    path('job-list.html', views.joblist_view, name='job_list'),
    path('job-detail.html', views.jobdetail_view, name='job_detail'),
    path('category.html', views.category_view, name='category'),
    path('testimonial.html', views.testimonial_view, name='testimonial'),
    path('404.html', views.error_404_view, name='error_404'),

    # API phát hiện và đếm phương tiện (POST request)
    path('detect-vehicles/', views.detect_vehicles_view, name='detect_vehicles'),

    # API đo tốc độ phương tiện (GET request)
    path('detect-vehicles-speed/', views.detect_vehicles_speed_view, name='detect_vehicles_speed'),

    # RESTful API đo tốc độ xe (cùng view, chỉ đường dẫn khác)
    path('api/vehicle-speed/', views.detect_vehicles_speed_view, name='vehicle_speed_api'),

    # API tính đường ngắn nhất (GET request)
    path('shortest-path/', views.shortest_path, name='shortest_path'),
]
