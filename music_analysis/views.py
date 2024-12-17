from .utils import processing_pipeline, get_inference
from django.shortcuts import render, get_object_or_404, redirect
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from django.urls import reverse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from itertools import combinations
from .forms import TuneForm
from .models import Tune
import plotly.colors
import numpy as np
import pandas as pd
import json


def index(request):
    """
    Displays a welcome message for the Irish Music Analyzer.

    Parameters:
    -----------
    request : HttpRequest
        The HTTP request object.

    Returns:
    --------
    HttpResponse
        A simple HTTP response with a welcome message.
    """
    return HttpResponse("Hello, welcome to the Irish Music Analyzer!")


def music_dashboard(request):
    """
    Renders the main music dashboard page.

    Parameters:
    -----------
    request : HttpRequest
        The HTTP request object.

    Returns:
    --------
    HttpResponse
        Renders the 'dashboard.html' template.
    """
    return render(request, 'dashboard.html')  # Make sure to create this template


def discover(request):
    """
    Renders the discover page for exploring musical features.

    Parameters:
    -----------
    request : HttpRequest
        The HTTP request object.

    Returns:
    --------
    HttpResponse
        Renders the 'discover.html' template.
    """
    return render(request, 'discover.html')


def about(request):
    """
    Renders the about page.

    Parameters:
    -----------
    request : HttpRequest
        The HTTP request object.

    Returns:
    --------
    HttpResponse
        Renders the 'about.html' template.
    """
    return render(request, 'about.html')


def tunes(request):
    """
    Manages the display, creation, updating, and deletion of tunes.

    Parameters:
    -----------
    request : HttpRequest
        The HTTP request object, containing POST data for creating or updating tunes,
        and optional 'edit' or 'delete' parameters in the query string.

    Returns:
    --------
    HttpResponse
        Renders the 'tunes.html' template, with context including:
            - 'tunes': List of all tunes.
            - 'form': TuneForm instance for creating or editing a tune.
            - 'tune_id': ID of the tune being edited, if any.
            - 'tunes_data_json': JSON-encoded data of all tunes.
            - 'delete_id': ID of the tune to be deleted, if any.
    """
    # List all tunes (Read)
    tunes = Tune.objects.all()

    # Check if a tune is being updated
    tune_id = request.GET.get('edit')  # Get 'edit' parameter from the query string
    delete_id = request.GET.get('delete')  # Get 'delete' parameter from the query string
    form = None

    # Debugging: Check if the request is a POST
    if request.method == 'POST':
        print("POST request received")

    # Handle Create/Update form
    if request.method == 'POST':
        if tune_id:
            # Update tune
            tune = get_object_or_404(Tune, pk=tune_id)
            form = TuneForm(request.POST, instance=tune)
        else:
            # Create new tune
            form = TuneForm(request.POST)

        if form.is_valid():
            form.save()
            return redirect('tunes')

    elif tune_id:
        # Populate form for editing
        tune = get_object_or_404(Tune, pk=tune_id)
        form = TuneForm(instance=tune)
    
    if delete_id and request.method == 'POST':
        # Debugging: Check if we are inside the delete logic
        print(f"Trying to delete tune with ID: {delete_id}")
        tune = get_object_or_404(Tune, pk=delete_id)
        tune.delete()
        print(f"Tune deleted: {delete_id}")
        return redirect('tunes')
    
    tunes_data = {}
    for tune in tunes:
        tunes_data[tune.pk] = {
            'name': tune.name,
            'composer': tune.composer,
            'abc_notation': tune.abc_notation,
        }

    return render(request, 'tunes.html', {
        'tunes': tunes,
        'form': form,
        'tune_id': tune_id,
        'tunes_data_json': json.dumps(tunes_data),
        'delete_id': delete_id
    })


def tunes_add(request):
    """
    Adds a new tune using a form, returning JSON for AJAX requests.

    Parameters:
    -----------
    request : HttpRequest
        The HTTP request object, expected to contain form data for POST requests.

    Returns:
    --------
    JsonResponse
        If the request is POST and form validation succeeds, returns JSON with success status.
        If validation fails, returns JSON with form HTML including error messages.
    HttpResponse
        Renders the add form template for GET requests.
    """
    if request.method == 'POST':
        form = TuneForm(request.POST)
        if form.is_valid():
            form.save()
            return JsonResponse({'success': True})
        else:
            # Render the form with errors
            html = render_to_string('tunes_form_partial.html', {'form': form}, request=request)
            return JsonResponse({'success': False, 'html': html})
    else:
        form = TuneForm()
    return render(request, 'tunes_form_partial.html', {'form': form})


def tunes_edit(request, pk):
    """
    Edits a specific tune by primary key (pk) using a form, returning JSON for AJAX requests.

    Parameters:
    -----------
    request : HttpRequest
        The HTTP request object, expected to contain form data for POST requests.
    pk : int
        The primary key of the tune to be edited.

    Returns:
    --------
    JsonResponse
        If the request is POST and form validation succeeds, returns JSON with success status.
        If validation fails, returns JSON with form HTML including error messages.
    HttpResponse
        Renders the edit form template for GET requests.
    """
    tune = get_object_or_404(Tune, pk=pk)
    if request.method == 'POST':
        form = TuneForm(request.POST, instance=tune)
        if form.is_valid():
            form.save()
            return JsonResponse({'success': True})
        else:
            # Render the form with errors
            html = render_to_string('tunes_form_partial.html', {'form': form, 'tune': tune}, request=request)
            return JsonResponse({'success': False, 'html': html})
    else:
        form = TuneForm(instance=tune)
    return render(request, 'tunes_form_partial.html', {'form': form, 'tune': tune})


def tunes_delete(request, pk):
    """
    Handles the deletion of a specific tune by primary key (pk).

    Parameters:
    -----------
    request : HttpRequest
        The HTTP request object, expected to be a POST request for deletion confirmation.
    pk : int
        The primary key of the tune to be deleted.

    Returns:
    --------
    JsonResponse
        A JSON response indicating success and the primary key of the deleted tune if the request method is POST.
    HttpResponse
        Renders a confirmation template if the request method is not POST.
    """
    print("Trying to delete tune")
    tune = get_object_or_404(Tune, pk=pk)
    if request.method == 'POST':
        tune.delete()
        return JsonResponse({'success': True, 'pk': pk})
    return render(request, 'tunes_confirm_delete_partial.html', {'tune': tune})


@csrf_exempt
def get_musical_features_data(request):
    """
    Retrieves musical features for tunes based on selected X, Y, and Z features, returning data for 3D plotting.

    Parameters:
    -----------
    request : HttpRequest
        The HTTP POST request containing JSON data with selected 'xFeature', 'yFeature', and 'zFeature' keys.

    Returns:
    --------
    JsonResponse
        A JSON response containing:
            - 'x': List of values for the selected X feature.
            - 'y': List of values for the selected Y feature.
            - 'z': List of values for the selected Z feature.
            - 'labels': List of tune names for labeling.
            - 'composerColorMapping': List of color mappings based on composer names.
    """
    if request.method == 'POST':
        # Parse the JSON body to get selected X, Y, and Z features
        body = json.loads(request.body)
        x_feature = body.get('xFeature')
        y_feature = body.get('yFeature')
        z_feature = body.get('zFeature')

        # Fetch all tunes
        tunes = Tune.objects.all()

        # Get all abc_notations
        abc_notations = [(tune.name, tune.composer, tune.abc_notation) for tune in tunes]

        # Pass to pipeline to run the KMeans algorithm
        tunes_extracted_features = processing_pipeline(abc_notations)

        composers = list(set(tune.composer for tune in tunes))
        all_colors = plotly.colors.qualitative.Plotly + plotly.colors.qualitative.Set1 + plotly.colors.qualitative.Set2
        
        # Composer Color Mapping
        composer_color_mapping={}
        color_index=0

        for current_composer in composers:
            while color_index > len(all_colors) - 1:
                color_index -= len(all_colors)
            composer_color_mapping[current_composer]= all_colors[color_index]
            color_index += 1

        
        # composer_color_mapping = {
        #     'Sean Ryan': 'red',
        #     'Paddy Fahey': 'yellow',
        #     'Lizz Carrol': 'green'
        # }

        # Initialize lists to hold the extracted features for X, Y, and Z axes
        x_data = []
        y_data = []
        z_data = []
        labels = []
        colors = []

        # Get selected features for each tune
        for tune_name, features in tunes_extracted_features.items():
            labels.append(tune_name)
            x_data.append(features.get(x_feature))
            y_data.append(features.get(y_feature))
            z_data.append(features.get(z_feature))
            colors.append(composer_color_mapping.get(features.get('composer')))

        # Prepare the response data
        return JsonResponse({
            'x': x_data,        # X-axis data
            'y': y_data,        # Y-axis data
            'z': z_data,        # Z-axis data
            'labels': labels,   # Tune names
            'composerColorMapping': colors
        })
    

def perform_clustering(request):
    """
    Performs k-means clustering on tunes based on selected features, returning cluster assignments and feature data.

    Parameters:
    -----------
    request : HttpRequest
        The HTTP POST request containing JSON data with selected 'xFeature', 'yFeature', and 'zFeature' keys.

    Returns:
    --------
    JsonResponse
        A JSON response containing:
            - 'x': List of values for the selected X feature.
            - 'y': List of values for the selected Y feature.
            - 'z': List of values for the selected Z feature.
            - 'clusters': List of cluster assignments for each tune.
            - 'composers': List of composer names for labeling data points.
    """
    if request.method == 'POST':
        # Parse the JSON body to get selected X, Y, and Z features
        body = json.loads(request.body)
        x_feature = body.get('xFeature')
        y_feature = body.get('yFeature')
        z_feature = body.get('zFeature')

        # Fetch all tunes
        tunes = Tune.objects.all()
        abc_notations = [(tune.name, tune.composer, tune.abc_notation) for tune in tunes]

        # Use the processing pipeline to extract features dynamically
        tunes_extracted_features = processing_pipeline(abc_notations)

        # Initialize lists to hold extracted features for clustering
        x_data = []
        y_data = []
        z_data = []
        composers = []

        # Get selected features for each tune
        for tune_name, features in tunes_extracted_features.items():
            x_data.append(features.get(x_feature))
            y_data.append(features.get(y_feature))
            z_data.append(features.get(z_feature))
            composers.append(features.get('composer'))

        # Prepare data for clustering (3D points based on selected features)
        features_data = np.array(list(zip(x_data, y_data, z_data)))

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=len(set(composers)), random_state=0)
        clusters = kmeans.fit_predict(features_data)

        all_colors = plotly.colors.qualitative.Plotly + plotly.colors.qualitative.Set1 + plotly.colors.qualitative.Set2
        current_colors = all_colors[:len(set(composers))]

        # Prepare the response data
        response_data = {
            'x': x_data,              # X-axis data
            'y': y_data,              # Y-axis data
            'z': z_data,              # Z-axis data
            'clusters': clusters.tolist(),  # Cluster assignment for each tune
            'composers': composers,          # Composer names for hover info
            'colorlist': current_colors      # Colors to plot with
        }

        return JsonResponse(response_data)
    

def calculate_feature_correlation(request):
    """
    Calculates the correlation matrix for numerical features extracted from all tunes, returning it as JSON.

    Parameters:
    -----------
    request : HttpRequest
        The HTTP POST request.

    Returns:
    --------
    JsonResponse
        A JSON response containing:
            - 'correlation_data': A dictionary representing the correlation matrix of numerical features.
            - 'feature_names': A list of feature names included in the correlation matrix for labeling.
    """
    if request.method == 'POST':
        # Fetch all tunes
        tunes = Tune.objects.all()
        abc_notations = [(tune.name, tune.composer, tune.abc_notation) for tune in tunes]

        # Use the processing pipeline to extract features for each tune
        tunes_extracted_features = processing_pipeline(abc_notations)

        # Create a DataFrame from the extracted features
        features_df = pd.DataFrame.from_dict(tunes_extracted_features, orient='index')

        # Exclude non-numeric columns (e.g., composer) from the correlation calculation
        numeric_features_df = features_df.select_dtypes(include=[float, int])

        # Calculate the correlation matrix and replace NaN values with 0
        correlation_matrix = numeric_features_df.corr().fillna(0)

        # Convert the correlation matrix to a format suitable for JSON response
        correlation_data = correlation_matrix.to_dict()

        return JsonResponse({
            'correlation_data': correlation_data,
            'feature_names': list(correlation_matrix.columns)  # Include feature names for labeling
        })
    

def search_tunes(request):
    """
    Searches for tunes by name or composer based on a query, returning results as JSON for AJAX requests.

    Parameters:
    -----------
    request : HttpRequest
        The HTTP GET request containing 'type' (search type) and 'query' parameters, and expected to be
        an AJAX request.

    Returns:
    --------
    JsonResponse
        A JSON response with a list of matching tunes, each containing its name, composer, edit URL,
        and delete URL.
    """
    if request.method == 'GET' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        search_type = request.GET.get('type')
        query = request.GET.get('query', '')

        # Filter tunes based on search type
        if search_type == 'name':
            tunes = Tune.objects.filter(name__icontains=query)
        elif search_type == 'composer':
            tunes = Tune.objects.filter(composer__icontains=query)
        else:
            tunes = Tune.objects.all()

        # Prepare the data to send back to the frontend
        tunes_data = [
            {
                'name': tune.name,
                'composer': tune.composer,
                'edit_url': reverse('tunes_edit', args=[tune.pk]),
                'delete_url': reverse('tunes_delete', args=[tune.pk])
            }
            for tune in tunes
        ]

        return JsonResponse({'tunes': tunes_data})
    

def test_tunes(request):
    """
    Renders the test tunes page.

    Parameters:
    -----------
    request : HttpRequest
        The HTTP request object.

    Returns:
    --------
    HttpResponse
        Renders the 'test_tunes.html' template.
    """
    return render(request, 'test_tunes.html')


def get_tune_feature_values(request):
    """
    Processes an ABC notation input and returns its calculated feature values as JSON.

    Parameters:
    -----------
    request : HttpRequest
        The HTTP POST request containing 'abc_notation' in the request data.

    Returns:
    --------
    JsonResponse
        A JSON response with calculated feature values for the given ABC notation,
        or an error message if 'abc_notation' is missing.
    """
    if request.method == 'POST':
        # Get ABC notation from the request
        abc_notation = request.POST.get('abc_notation')

        if not abc_notation:
            return JsonResponse({'error': 'No ABC notation provided'}, status=400)

        # Process the ABC notation using processing_pipeline
        features = processing_pipeline([('unknown', 'unknown', abc_notation)])
        del features['unknown']['composer']
        
        # Return the feature values as JSON
        return JsonResponse(features['unknown'])
    

def get_tune_comparisons(request):
    """
    Retrieves musical feature comparisons for stored tunes and a user-uploaded ABC notation, if provided.
    Calculates feature values for each tune and organizes them into data for 3D scatter plots, based on 
    unique feature triplets.

    Parameters:
    -----------
    request : HttpRequest
        The HTTP request object containing an optional 'abc_notation' parameter for user-uploaded ABC notation.

    Returns:
    --------
    JsonResponse
        A JSON response with:
            - 'features': A dictionary of calculated feature values for the user-uploaded ABC notation.
            - 'plots': A dictionary with plot data for each feature triplet, containing 'x', 'y', 'z' values,
              'labels' for tune names, and 'composer' names for coloring points in the plot.
    """
    abc_notation = request.GET.get('abc_notation', None)
    # List all features to calculate
    features = ["notes", "rests", "chords", "avg_pitch", "duration_sd"]
    feature_triplets = list(combinations(features, 3))

    # Process tunes
    tunes = Tune.objects.all()
    abc_notations = [(tune.name, tune.composer, tune.abc_notation) for tune in tunes]
    tunes_features = processing_pipeline(abc_notations)

    # Process uploaded tune features if abc_notation is provided
    uploaded_tune_features = None
    if abc_notation:
        uploaded_tune_features = processing_pipeline([('UserSubmitted', 'User', abc_notation)])['UserSubmitted']

    # Prepare data for plots
    data_for_triplets = {}
    for triplet in feature_triplets:
        triplet_data = {
            'x': [features.get(triplet[0], 0) for features in tunes_features.values()],
            'y': [features.get(triplet[1], 0) for features in tunes_features.values()],
            'z': [features.get(triplet[2], 0) for features in tunes_features.values()],
            'labels': [name for name in tunes_features.keys()],
            'composer': [features.get("composer", "unknown") for features in tunes_features.values()]
        }

        # Add uploaded tune's features to plot data if available
        if uploaded_tune_features:
            triplet_data['x'].append(uploaded_tune_features[triplet[0]])
            triplet_data['y'].append(uploaded_tune_features[triplet[1]])
            triplet_data['z'].append(uploaded_tune_features[triplet[2]])
            triplet_data['labels'].append("UserSubmitted")
            triplet_data['composer'].append("User")

        data_for_triplets[", ".join(triplet)] = triplet_data

    # Return both features and plots in response
    return JsonResponse({
        'features': uploaded_tune_features if uploaded_tune_features else {},
        'plots': data_for_triplets
    })


def make_inference(request):
    """
    Handle a POST request to classify an ABC notation as a composer.

    Args:
        request: Django HTTP request containing 'abc_notation' in POST data.

    Returns:
        JsonResponse: JSON containing the predicted composer or an error message.
    """
    print("Making inference")
    if request.method == "POST":
        abc_notation = request.POST.get("abc_notation", "").strip()

        if not abc_notation:
            return JsonResponse({"error": "No ABC notation provided"}, status=400)

        try:
            composer_name = get_inference(abc_notation)
            print(f"Predicted composer: {composer_name}")
            return JsonResponse({"composer": composer_name})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)


def perform_clustering_2(request):
    """
    Perform PCA on extracted features and apply K-means clustering, including user-uploaded tune.
    """
    # Get user-uploaded ABC notation (if any)
    user_abc_notation = request.GET.get('abc_notation', None)

    # Extract tunes and process features
    tunes = Tune.objects.all()
    features = processing_pipeline([(tune.name, tune.composer, tune.abc_notation) for tune in tunes])

    # Prepare data for PCA
    feature_names = ['notes', 'rests', 'chords', 'avg_pitch', 'pitch_range', 'duration_sd', 'pitches_len', 'avg_duration',
                     'duration_range', 'duration_sd', 'total_duration', 'avg_interval', 'interval_range', 'interval_sd']    
    data_matrix = []
    composers = []
    labels = []
    for tune_name, feature_data in features.items():
        row = [feature_data[feature] for feature in feature_names]
        data_matrix.append(row)
        composers.append(feature_data['composer'])
        labels.append(tune_name)

    user_index = None
    # Process the user-uploaded ABC notation
    if user_abc_notation:
        user_features = processing_pipeline([('UserSubmitted', 'User', user_abc_notation)])['UserSubmitted']
        user_row = [user_features[feature] for feature in feature_names]
        data_matrix.append(user_row)
        composers.append('User')
        labels.append('UserSubmitted')
        user_index = len(labels) - 1  # Store index of user-uploaded tune

    # Convert to NumPy array
    data_matrix = np.array(data_matrix)

    # Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(data_matrix)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=9, random_state=0)
    clusters = kmeans.fit_predict(pca_features)

    # Prepare response data
    return JsonResponse({
        'x': pca_features[:, 0].tolist(),
        'y': pca_features[:, 1].tolist(),
        'z': pca_features[:, 2].tolist(),
        'clusters': clusters.tolist(),
        'composers': composers,
        'labels': labels,
        'user_index': user_index,  # Explicitly mark the user-uploaded tune
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist()  # For debugging/analysis
    })

# def perform_clustering_2(request):
#     if request.method == 'POST':
#         # Parse JSON to get features
#         body = json.loads(request.body)
#         abc_notation = body.get('abc_notation')

#         # Fetch all tunes
#         tunes = Tune.objects.all()
#         abc_notations = [(tune.name, tune.composer, tune.abc_notation) for tune in tunes]

#         # Extract features using processing_pipeline
#         features = processing_pipeline(abc_notations)

#         # Add user-uploaded tune features
#         uploaded_features = processing_pipeline([('UserUploaded', 'User', abc_notation)])['UserUploaded']

#         # Create dataset for clustering
#         X = np.array([
#             [feat['avg_pitch'], feat['pitch_range'], feat['duration_sd']]
#             for feat in features.values()
#         ])
#         X = np.vstack((X, [
#             uploaded_features['avg_pitch'],
#             uploaded_features['pitch_range'],
#             uploaded_features['duration_sd'],
#             uploaded_features['notes'],
#             uploaded_features['rests'],
#             uploaded_features['chords'],
#             uploaded_features['pitches_len'],
#             uploaded_features['avg_duration'],
#             uploaded_features['duration_range'],
#             uploaded_features['total_duration'],
#             uploaded_features['avg_interval'],
#             uploaded_features['interval_range'],
#             uploaded_features['interval_sd'],
#             uploaded_features['duration_sd'],
#         ]))  # Add the user-uploaded tune to the dataset

#         # Perform PCA to reduce dimensions
#         pca = PCA(n_components=3)
#         X_pca = pca.fit_transform(X)

#         # Perform KMeans clustering
#         kmeans = KMeans(n_clusters=9, random_state=0)
#         clusters = kmeans.fit_predict(X_pca)

#         # Find distances between user-uploaded tune and cluster centroids
#         user_point = X_pca[-1]
#         distances = pairwise_distances([user_point], kmeans.cluster_centers_).flatten()

#         # Prepare response data
#         response_data = {
#             'x': X_pca[:, 0].tolist(),
#             'y': X_pca[:, 1].tolist(),
#             'z': X_pca[:, 2].tolist(),
#             'clusters': clusters.tolist(),
#             'composers': [*features.keys(), 'User'],
#             'distances': distances.tolist(),  # Distances to each cluster centroid
#             'cluster_labels': [f'Cluster {i}' for i in range(9)]
#         }

#         return JsonResponse(response_data)