# SN
Image-based construction and analysis of stromatolite spatial networks.

The workflow is designed to investigate how the spatial organization of stromatolites may influence their local environment and collective spatial structure. The project extracts individual stromatolites from top-view image, characterizes the empty space between them using Delaunay triangulation, constructs spatial networks from their geometric relationships, and analyzes the resulting network properties.

<img width="4000" height="2250" alt="GA" src="https://github.com/user-attachments/assets/2d4130ec-7450-47d0-862c-9e0613019bdf" />

#### Main Steps
**1. Image preprocessing**
   - Image grayscale conversion
   - Gaussian filtering
   - Otsu thresholding
   - Morphological processing
   - Distance transform and foreground extraction

**2. stromatolite segmentation.**
   - Watershed segmentation
   - Extraction of individual stromatolite contours
   - Calculate geometric properties such as surface area, perimeter, and centroid

**3. Empty-space partitioning**
   - Delaunay triangulation of stromatolite contour points
   - Identification of triangles located outside stromatolite structures
   - Characterization of the spatial regions between neighbouring stromatolites

**4. Spatial network construction**
   - Stromatolites are represented as network nodes
   - Connections between neighbouring stromatolites are derived from the Delaunay representation
   - Different edge-weight definitions can be used to describe spatial relationships between stromatolites

**5. Network analysis**
   - Degree (`k`)
   - Strength (`s`)
   - Edge weight (<code>w<sub>ij</sub></code>)
   - Neighborhood Watch (`Nw = s/k`)
   - Area-normalised and perimeter-normalised network metrics
   - Power-law scaling relationships between geometric and network properties


*For requirement of more data and code, please contact zhouziyun1900@hotmail.com*  
