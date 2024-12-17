from django.urls import path
from . import views

urlpatterns = [
    path('test', views.index, name='index'),
    path('', views.music_dashboard, name='music_dashboard'),
    path('discover/', views.discover, name='discover'),
    path('tunes/', views.tunes, name='tunes'),
    path('test-tunes/', views.test_tunes, name='test-tunes'),
    path('tunes/add/', views.tunes_add, name='tunes_add'),
    path('tunes/edit/<int:pk>/', views.tunes_edit, name='tunes_edit'),
    path('tunes/delete/<int:pk>/', views.tunes_delete, name='tunes_delete'),
    path('get-musical-features/', views.get_musical_features_data, name='get_musical_features_data'),
    path('perform-clustering/', views.perform_clustering, name='perform_clustering'),
    path('calculate-feature-correlation/', views.calculate_feature_correlation, name='calculate_feature_correlation'),
    path('search-tunes/', views.search_tunes, name='search_tunes'),
    path('calculate-features/', views.get_tune_feature_values, name='calculate_features'),
    path('get-tune-comparisons/', views.get_tune_comparisons, name='get_tune_comparisons'),
    path('make-inference/', views.make_inference, name='make_inference'),
    path('about/', views.about, name='about'),
    path('perform-clustering-2/', views.perform_clustering_2, name='perform-clustering-2'),
]
