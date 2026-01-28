# QField Manual: DWA-M-921 Erosion Survey

**Field Guide for Soil Erosion Mapping**

---

**Authors:**
- Edberto Moura Lima (BAW Research) - [ORCID](https://orcid.org/0000-0002-8447-8460)
- Gunther Liebhard (BAW Research)
- Thomas Brunner (BAW-IKT)
- Peter Strauss (BAW-IKT)

**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

> **Language Selection / Sprachauswahl:** **English (current)** | [Deutsche Version](Manual_de.md)

---

## Table of Contents

- [1. Introduction](#1-introduction)
  - [1.1 Scope](#11-scope)
- [2. Getting Started](#2-getting-started)
  - [2.1 About QField](#21-about-qfield)
  - [2.2 Project Setup](#22-project-setup)
  - [2.3 Opening the Project](#23-opening-the-project)
- [3. User Interface](#3-user-interface)
  - [3.1 Map Canvas](#31-map-canvas)
  - [3.2 Dashboard and Legend](#32-dashboard-and-legend)
- [4. Field Mapping Workflow](#4-field-mapping-workflow)
  - [4.1 Survey Concepts](#41-survey-concepts)
  - [4.2 Basic Editing](#42-basic-editing)
  - [4.3 Digitizing a Survey Unit](#43-digitizing-a-survey-unit)
  - [4.4 Recording Erosion Features](#44-recording-erosion-features)
  - [4.5 Attaching Photographs](#45-attaching-photographs)
- [5. Advanced Features](#5-advanced-features)
  - [5.1 Geometry Editing](#51-geometry-editing)
  - [5.2 Deleting Features](#52-deleting-features)
  - [5.3 Additional Settings](#53-additional-settings)
- [6. Data Management](#6-data-management)
  - [6.1 Data Export and Synchronization](#61-data-export-and-synchronization)
- [Appendices](#appendices)
  - [Appendix A: Overview Mapping Parameters](#appendix-a-overview-mapping-parameters)
  - [Appendix B: Detailed Mapping Parameters](#appendix-b-detailed-mapping-parameters)
  - [Appendix C: Additional Mapping Parameters](#appendix-c-additional-mapping-parameters)

---

## 1. Introduction

These tutorials guide you through the Bodenerosion durch Wasser workflow to set up a data-collection project, map features in the field, capture information using digital forms, share the project with a team of collaborators, visualise project data through web maps and dashboards, and automate reporting tasks based on field observations.

The tutorials are based on a hypothetical survey at the LTE HOAL in Petzenkirchen. However, the concepts and skills presented are transferable to a wide range of data-collection scenarios. As such, erosion surveys, baseline and project monitoring and evaluation, ground truth data generation for remote sensing and soil erosion classification, or detailed GIS mapping. This mapping guide serves as a practical tool for documenting all erosion subprocesses, from detachment and transport to deposition, by placing them in their ecosystem context and describing them using standardized terminology.

Depending on the scale and intensity of the mapping, a distinction is made between overview and detailed mapping. The mapping guide specifically refers to soil erosion by water, which mostly occurs on arable land.

The methodology applied in this project follows established standards, specifically the guidelines of the German Water Association (DWA) for documenting water-induced soil erosion, known as **DWA-M-921** (Bodenerosion durch Wasser – Kartieranleitung zur Erfassung aktueller Erosionsformen).

### 1.1 Scope

This mapping guide is limited to the documentation of water-induced soil erosion on arable land. It excludes other erosion processes and soil displacement mechanisms, including wind erosion, subterranean tunnel erosion, and erosion occurring in forests, grasslands, ski slopes, embankments, or on unpaved roads. Special erosion forms such as snow erosion and tillage-induced erosion caused by agricultural machinery are also excluded.

> **Note:** This guide is restricted to the mapping and documentation of erosion features. It does not include procedures for damage assessment, the design of mitigation or protection measures, or the evaluation of future erosion risk.

---

## 2. Getting Started

### 2.1 About QField

**QField** is a mobile GIS application built to be user-friendly and intuitive. It allows field workers to collect, edit, and manage geospatial data directly in the field using smartphones or tablets. This digital approach ensures efficient data capture and seamless integration with the project's GIS infrastructure.

#### Where can I get QField?

The mobile application is provided by the Swiss company [OPENGIS](https://qfield.org/) and is available free of charge in the Google Play Store. The software is released under the GNU General Public License version 2.0 (GPL-2.0), allowing use, modification, and redistribution in accordance with the license terms. The creation and preparation of project-specific datasets require the use of the desktop GIS software [QGIS (version 3 or later)](https://qgis.org/).

OPENGIS provides an optional cloud service that enables centralized project management and data synchronization via the internet ([QFieldCloud](https://qfield.cloud/)). The application can also be fully operated without the use of this service.

#### Further reading

Explore the official QField docs for getting started with [QFieldCloud](https://docs.qfield.org/get-started/tutorials/get-started-qfc/) and [QFieldSync](https://docs.qfield.org/get-started/tutorials/get-started-qfs/). The [QField How-to guides](https://docs.qfield.org/how-to/) have lots of tips and examples for creating great data collection projects.

### 2.2 Project Setup

On the QField home screen users are presented with two options to open a project:

- **QFieldCloud projects**: Access a project stored on QFieldCloud (cloud-based synchronization)
- **Open local file**: Copy a working copy of the QGIS project file from a laptop or PC to the phone or tablet

![QField Application homescreen](../screemshots/Screenshot_20260127_101009.jpg)

For this tutorial, we will use **Open local file** to import the Bodenerosion durch Wasser project.

> **Note:**
> - The app can be used in the field without an internet connection.
> - An internet connection is required to download the app and the associated file.
> - GPS must be switched on during field data collection.

#### Import from Local Storage

After selecting "Open local file", you will see the file browser interface with three main directories:

- **QField files directory**
- **Imported datasets**
- **Imported projects**

A drop-down menu accessible via a bottom-right plus button lists the means to import projects and datasets:

- **Import project from folder**
- **Import project from ZIP** (archive)
- **Import (individual) dataset(s)**
- **Import from URL**

![Import menu from local file](../screemshots/Screenshot_20260127_101551.jpg)

#### Import Project from Folder or ZIP Archive

When importing a project from a folder or a ZIP archive:

1. You will be asked to grant permission for QField to read the content of a given folder on the device's storage via a system folder picker
2. When the folder or the archive is selected, QField copies the content (including its sub-folders) into the app's "Imported projects" location
3. You can then open the project from there

> **Tip:** Re-importing a given folder through the drop-down menu action will overwrite preexisting projects given an identical folder name. This allows you to update projects.

#### Import Individual Datasets

You can also import individual datasets:

1. You will be asked to select one or more files via a system file picker
2. Files will be copied into the "Imported datasets" folder
3. Ensure that all sidecar files are selected when importing (e.g., a Shapefile dataset would require you to select the .shp, .shx, .dbf, .prj, and .cpg files)

#### Import from URL (Recommended)

When importing a project or individual dataset through the "Import URL" action, you will be asked to provide a URL string to a file. QField will subsequently fetch the content and save it into the "Imported projects" or "Imported datasets", respectively.

![Importing from a URL](../screemshots/Screenshot_20260127_102013.jpg)

This is the easiest method to get the Bodenerosion durch Wasser project:

1. Tap **"Import from URL"**
2. Enter the following URL:

```
https://github.com/EdbertoLima/BodenerosionKartieranleitung/raw/main/erosion_survey_qfield.zip
```

3. Tap **"OK"** to start the download
4. Wait for QField to download and extract the project
5. The project appears in "Imported projects" when complete

> **Caution:** Re-importing a project folder with the same name will overwrite the existing project. Use this method to update projects with new configurations or empty databases before fieldwork.

### 2.3 Opening the Project

1. Navigate to **"Imported projects"** in the file browser
2. Locate the **erosion_survey** folder
3. Tap on `erosion_survey.qgz` to open the project
4. Wait for all layers to load

---

## 3. User Interface

### 3.1 Map Canvas

After opening a project, the central area of the screen displays the map with all loaded layers.

![Map view showing the survey area](../screemshots/Screenshot_20260127_114135.jpg)

You can interact with the map using touch gestures:

- **Pan**: Drag with one finger
- **Zoom**: Pinch with two fingers
- **Rotate**: Two-finger rotation gesture

### 3.2 Dashboard and Legend

The Dashboard provides access to layers, settings, and Digitize Mode.

**To open the Dashboard:**

1. Tap the **menu icon** (☰) in the top-left corner, OR
2. Swipe from the **left edge** of the screen toward the center

Open the side "Dashboard" and expand the layers list to display the legend of the map.

![Dashboard panel](../screemshots/Screenshot_20260127_102223.jpg)

#### Layer Options

Long-press on a layer name to access these options:

| Action | Description |
|--------|-------------|
| **Expand legend item** | Show/hide the layer's sub-items |
| **Show on map** | Control visibility |
| **Show labels** | Control the visibility of the labels |
| **Opacity Slider** | Control the transparency of the layer |
| **Zoom to layer** | Have all the layer items on the map |
| **Reload icon** | Get the current data of a layer with remote sources |
| **Show feature list** | Show all the layer's features in the identification list |
| **Setup tracking** | Set up tracking mode of layer |

---

## 4. Field Mapping Workflow

### 4.1 Survey Concepts

The project organizes data into specific "layers" based on the shape of the feature you are mapping. This structure follows the **DWA-M-921** standard, which requires separating erosion features into Areas (Polygons), Lines, and Points.

Think of these layers as digital containers. You must select the correct container before you start drawing:

| **If you see this in the field...** | **Use this Layer** | **Geometry** |
|-------------------------------------|-------------------|--------------|
| **Field Boundaries** - The perimeter of the parcel or field block. | `Flächen-/Feldblock` | Polygon ⬠ |
| **Large Erosion Areas** - Sheet erosion, sediment fans, or land use patterns. | `Flächenhafte` | Polygon ⬠ |
| **Channels & Flow Paths** - Rills, gullies, drainage ditches, or flow paths. | `Linienhafte` | Line ∿ |
| **Specific Spots** - Pipe outlets, headcuts, damage points, small deposits. | `Punktförmige` | Point • |

#### Mapping Intensity Levels

Before you begin recording attributes, you must decide on the "intensity" of your survey. You specify this choice within the **Field Block** (*Flächen-/Feldblock*) attributes. The choice of mapping intensity depends on the scale and the specific question you need to answer.

The DWA-M-921 standard distinguishes between two mapping approaches:

##### 1. Overview Mapping (*Übersichtskartierung*)

- **Goal:** A generalized representation of erosion events, causes, and effects.
- **Method:** Focuses on qualitative descriptions (e.g., "Severe rill erosion present") rather than precise measurements.
- **Workflow:** Often performed as a "Windshield Survey" from the field edge.
- **Use Case:**
  - Identifying damage hotspots and causes within a larger catchment area.
  - Long-term monitoring of large-scale erosion trends.
  - Visualizing erosion systems (transport paths, accumulation, and off-site effects).
  - Planning locations for future detailed survey.

##### 2. Detailed Mapping (*Detailkartierung*)

- **Goal:** Precise location and quantification of soil loss.
- **Method:** Requires measuring the exact dimensions of every feature (width, depth, length) to calculate volume in cubic meters.
- **Workflow:** Requires walking the entire field and accessing every feature on-site.
- **Specific Tasks:**
  - Transfer Points (Übertrittstellen): Characterize exactly where water and sediment enter ditches, water bodies, neighboring fields, or infrastructure (roads/pipes).
  - Site Conditions: Describe the catchment size, slope, soil properties, management methods, crops, and existing soil protection measures.
  - Landscape Elements: Map features that inhibit erosion (e.g., hedges, grass buffer strips).
  - Quantification: Exact measurement of erosion and accumulation forms.

> **Important:** All parameters recorded in Overview Mapping are **also mandatory** for Detailed Mapping. Detailed mapping simply adds more fields (like volume and exact dimensions) on top of the basic overview data. See the Appendices for the full parameter lists.

### 4.2 Basic Editing

QField operates in two distinct modes:

| **Mode** | **Icon** | **Function** | **Think of it as...** |
|----------|----------|--------------|----------------------|
| **Browse Mode** | (Default) | **View & Inspect.** Tap features to see their data, but cannot move or create them. | **"Reading"** mode. |
| **Digitize Mode** | ✎ (Pencil) | **Create & Edit.** Unlocks the map for drawing new lines, points, and polygons. | **"Writing"** mode. |

#### Browse Mode

As the name suggests, while being in browse mode, you can view and select features within all identifiable layers in the project. It is also possible to edit attributes of existing features by clicking on a feature of interest and opening its attribute table.

#### How to Start Digitizing

To record a new erosion feature, you must actively switch the app into "Writing" mode.

1. **Open Dashboard:** Tap the **Menu (☰)** or swipe from the left.
2. **Enable Editing:** Tap the **Pencil Icon** ✎. The interface will change.
3. **Select Layer:** Tap the layer you want to edit (e.g., `Linienhafte` for a rill).
   - *Visual Check:* The active layer name will turn **GREEN**.
4. **Draw:** Close the dashboard and use the **(+)** button to drop points on the map.

> **Tip:** QField ensures that digitized geometries will not have duplicate vertices and respects the geometry precision settings from the currently selected layer.

### 4.3 Digitizing a Survey Unit

A **survey unit** (Flächen-/Feldblock) is required for each mapping entry. This parcel is defined using the `Flächen-/Feldblock` layer and serves as the spatial reference for the assessment. It must be completed before any erosion features can be recorded. The level of detail (Overview vs. Detailed Mapping) is specified within the parcel's attributes.

#### Step 1: Select Layer and Enable Digitizing

1. Open the **Dashboard** (swipe from left or tap ☰) and Tap the **pencil icon** to enable Digitize Mode.
2. Select the `Flächen-/Feldblock` layer (highlighted in green).

![Enabling Digitize Mode](../screemshots/Screenshot_20260127_102334.jpg)

#### Step 2: Digitize the Parcel Boundary

1. Position the crosshair at a field corner and tap **[+]** to add a vertex.
2. Repeat for all corners to trace the perimeter.
3. Tap **[✓]** to complete the polygon

![Adding vertices](../screemshots/Screenshot_20260127_102649.jpg)

#### Step 3: Complete Attributes

1. The attribute form opens automatically.
2. Fill in the required survey data (Identifier, Date, Land Use).
3. Tap **[✓]** to store the parcel.

![Field block attribute form](../screemshots/Screenshot_20260127_102914.jpg)

### 4.4 Recording Erosion Features

#### Recording Polygon Features

To record Polygon (area features) features use the `Flächenhafte` layer. This layer is intended for mapping areal and areal-linear erosion forms, accumulation areas, catchment areas of erosion systems, as well as land-use and management types.

**Steps:**

1. Open the **Dashboard** (☰) and tap the **pencil icon** to enable Digitize Mode.
2. Select the `Flächenhafte` layer.
3. Navigate the crosshair in the center of the screen to the desired start of the polygon
4. Click the **Plus (+)** button at the lower right of the screen to add the first node
5. Add more points to form your polygon by clicking the **Plus (+)** button each time you want to add a new node
6. (Optional) Click the **Minus (-)** button to remove the last added node
7. Click on the **[✓]** button to finish your edition (minimum 3 nodes for polygons)
8. (Optional) You can click the **(x)** button to cancel the current feature creation
9. Complete the attribute form and tap **[✓]** to store the feature

#### Recording Linear Features

Use the `Linienhafte` layer to map rills, gullies, ephemeral gullies, and concentrated flow paths. Linear features are digitized along the centerline of the erosion channel.

> **Important:** Assess the erosion feature before digitizing:
> - Locate the **starting point** (usually upslope, at the headcut)
> - Follow the **channel path** to the endpoint
> - Note the **width** and **depth** variations
> - Identify any **branches** or **confluences**

**Steps:**

1. Open the **Dashboard** (☰) and tap the **pencil icon** to enable Digitize Mode.
2. Select the `Linienhafte` layer.
3. Position the crosshair at the start of the erosion feature (usually upslope at the headcut)
4. Click the **Plus (+)** to add the first node
5. Add more points to form your line by clicking the **Plus (+)** button each time you want to add a new node
6. Click on **[✓]** to complete the line (minimum 2 vertices)
7. Complete the attribute form and tap **[✓]** to store the feature

> **Tip:** Always digitize linear erosion features from **upslope to downslope** (head to outlet). This convention ensures consistent data analysis and flow direction interpretation.

#### Recording Point Features

Use the `Punktförmige` layer to record pipe outlets, headcuts, specific damage points, and photo documentation locations.

**Steps:**

1. Open the Dashboard (☰) and tap the pencil icon to enable Digitize Mode
2. Select the **`Punktförmige`** layer
3. Close the Dashboard
4. Navigate to the feature location
5. Position the crosshair **exactly** at the point of interest
6. Tap **[✓]** to place the point
7. Complete the attribute form and tap **[✓]** to store the feature

![Placing a point feature](../screemshots/Screenshot_20260127_104830.jpg)

### 4.5 Attaching Photographs

Photographs provide essential visual documentation of erosion features. QField allows you to capture and attach photos directly to feature records.

#### Step 1: Locate the Photo Field

1. After creating a feature geometry, the attribute form opens
2. Scroll through the form to find the **Photo** or **Attachments** field
3. The field displays a **camera icon** or **attachment button**

#### Step 2: Capture a New Photo

1. Tap the **camera icon** or **"Take Photo"** button
2. The device camera application opens
3. Frame the feature of interests in the viewfinder:
   - Include **context** (surrounding area)
   - Show **scale reference** (ruler, person, object)
   - Capture **detail** of erosion characteristics
4. Tap the **shutter button** to take the photo
5. Review the image
6. Tap **"OK"**, **"Use Photo"**, or **checkmark** to accept
7. The photo is attached to the feature record

![Camera interface](../screemshots/Screenshot_20260127_103806.jpg)

#### Step 3: Add Multiple Photos

Capture additional photos showing:
- **Overview** — Wide shot showing feature in landscape context
- **Detail** — Close-up of erosion characteristics
- **Cross-section** — Channel shape and depth
- **Upstream/Downstream** — Views along the feature

#### Step 4: Add Existing Photos from Gallery

1. Tap the **gallery icon** or **"Add from Gallery"** option
2. Browse to select existing photos
3. Photos are attached to the feature record

> **Photo Best Practices:**
> - **Always include a scale reference** (ruler, measuring tape, or known object)
> - **Capture multiple angles** for comprehensive documentation
> - **Note the photo direction** in comments (e.g., "Looking downslope")
> - **Ensure adequate lighting** for clear detail visibility
> - **Avoid shadows** that obscure erosion features

---

## 5. Advanced Features

### 5.1 Geometry Editing

To edit the geometry of pre-existing features:

1. Enable the **Digitize mode** by tapping on the pencil icon underneath the legend
2. Once in digitize mode, a new **Edit geometry** button will appear in the title bar of an identified feature form
3. Clicking on the button will activate the geometry editing environment

#### Editing Tools

##### Vertex Tool

The vertex editor allows you to move, delete, or add new vertices to geometries.

![Editing vertices](../screemshots/Screenshot_20260127_103035.jpg)

##### Reshape Eraser Tool

The reshape eraser tool is designed to ease the removal of parts of a line or polygon geometry. The tool mimics eraser tools from 2D drawing programs and works best with a stylus.

### 5.2 Deleting Features

Deleting a feature is done by:

1. Tap on the feature to identify it
2. Open the feature form
3. Select the **Delete feature** action in the feature form's (⋮) menu
4. Confirm deletion

![Delete feature option](../screemshots/Screenshot_20260127_104903.jpg)

#### Delete Multiple Features

QField also allows you to delete multiple features at a time:

1. First identify the features by short tapping on the relevant parts of the map
2. Activate the multi-selection mode by **long pressing** on one of the features you want to delete
3. When checkboxes appear next to the feature names, select further features to delete
4. Once done, select the **Delete Selected Feature(s)** action in the features list (⋮) menu
5. Confirm deletion

> **Warning:** Deletion is permanent. Ensure you have a backup before deleting features.

### 5.3 Additional Settings

There are other more advanced settings which you can enable to make your data collection more efficient:

- **Use volume keys to digitize**: If you want to avoid having to tap on your device for every node, you can enable this option to add and remove nodes using the volume keys (Android only)
- **Allow finger tap on canvas to add vertices**: If you want to use your finger to add nodes as well rather than having to press the button

To enable both options:

1. Open the side "Dashboard" Panel (☰)
2. Navigate to **Settings > General**

#### Remember Attribute Values

For quick collection of rather homogeneous datasets, it is crucial to not have to enter the same attribute values multiple times. The pins on the right of every attribute enable the **last entered value** option for each attribute individually so that the next time you add a feature on the same layer, these attributes will be automatically pre-filled.

> **Note:** This last entered value only applies when collecting new features, not when editing existing ones.

---

## 6. Data Management

### 6.1 Data Export and Synchronization

Once you are done with your fieldwork, there are several ways to send and export the changed files back to the source device.

#### Export to Folder

When choosing the "Export to folder" action:

1. You will be asked to pick a location where the content will be copied to
2. You can use this action to copy the content of modified projects or datasets into a folder on your device that can be accessed by third-party synchronization apps such as Syncthing
3. You can also directly copy content into cloud accounts of providers that support Android's Scoped Storage directory provider (e.g., NextCloud)

#### Send Compressed Project Folder

The "Send compressed folder to" action:

1. Compresses the content of a selected folder into a ZIP archive
2. You will be asked through which app the resulting ZIP archive should be sent

You can compress and send whole projects by selecting root folders in QField's "Imported projects" directory, as well as send selective folders within project folders (for instance, your photos only).

![Export to folder option](../screemshots/Screenshot_20260127_105348.jpg)

#### Synchronization Options

| Method | Description |
|--------|-------------|
| **Direct cable transfer** | Connect device to PC |
| **Cloud sync** | NextCloud, Google Drive, Syncthing |
| **Email** | Send compressed project |
| **QFieldCloud** | Cloud-based synchronization service |

---

## Appendices

### Appendix A: Overview Mapping Parameters

The objective of overview mapping is to record and document the dominant erosion features occurring in the field. The focus is on identifying erosion events and their main characteristics rather than on a detailed quantitative assessment.

| Lfd. Nr. | Attribut / Kartierobjekt |
|---------:|--------------------------|
| **0.** | **Allgemeine Daten zur Kartierung** |
| 0.1 | Identifikator für die Kartierung / das kartierte Ereignis |
| 0.2 | Kartierdatum |
| 0.3 | Datum des Erosionsereignisses |
| 0.4 | Andauer des Erosionsereignisses |
| 0.5 | Nummer des Erosionssystems (Objekt-Identifikator) |
| 0.6 | Flächen- / Feldblock-Identifikator |
| 0.7 | Fotos des Erosionssystems (Identifikationsnummer der Fotos) |
| 0.8.1 | Name der/des Kartierenden |
| 0.8.2 | Institution / Dienststelle der/des Kartierenden |
| **1.** | **Flächennutzung, Bodenbearbeitung und Bewirtschaftung** |
| 1.1 | Nutzungsart / Nutzungsform |
| 1.2 | Angebaut(e) Fruchtart(en) |
| 1.3 | Bearbeitungszustand der Ackerfläche |
| 1.5.3 | Bedeckung durch Pflanzen und Pflanzenreste (in %) |
| **2.** | **Abtrags- und Akkumulationsformen** |
| 2.1.1 | Flächenhafte Abtragsformen |
| 2.1.2 | Lineare Abtragsformen |
| 2.1.2.6 | Stärke des linearen Abtrags |
| 2.1.5.4 | Erosion in Mulden und Tiefenlinien |
| 2.2 | Akkumulationen |
| **3.** | **Übertritte von erodiertem Boden, flächenexterne Wirkungen („off-site")** |
| 3.1 | Übertrittsstellen |
| **4.** | **Oberflächenwasser, Fremdwasserzufluss und Wasseraustritt** |
| 4.1 | Fremdwasserzufluss (allgemein) |
| **5.** | **Relief, Reliefmerkmale** |
| 5.2 | Hangneigung im Bereich der Abtragsform |

### Appendix B: Detailed Mapping Parameters

Detailed mapping follows the same workflow as overview mapping. All parameters recorded during overview mapping are also **mandatory for detailed mapping**. The main difference is the higher level of detail, particularly in the more precise classification, measurement, and quantification of erosion and accumulation forms.

| Lfd. Nr. | Attribut / Kartierobjekt |
|---------:|--------------------------|
| **0.** | **Allgemeine Daten zur Kartierung** |
| 0.1 | Identifikator für die Kartierung / das kartierte Ereignis |
| 0.2 | Kartierdatum |
| 0.3 | Datum des Erosionsereignisses |
| 0.4 | Andauer des Erosionsereignisses |
| 0.5 | Nummer des Erosionssystems (Objekt-Identifikator) |
| 0.6 | Flächen- / Feldblock-Identifikator |
| 0.6.1 | Flächengröße der kartierten Parzelle (in ha) |
| 0.8.1 | Name der/des Kartierenden |
| 0.8.2 | Institution / Dienststelle der/des Kartierenden |
| **1.** | **Flächennutzung, Bodenbearbeitung und Bewirtschaftung** |
| 1.1 | Nutzungsart / Nutzungsform |
| 1.2 | Angebaut(e) Fruchtart(en) |
| 1.3 | Bearbeitungszustand der Ackerfläche |
| 1.5.3 | Bedeckung durch Pflanzen und Pflanzenreste (in %) |
| **2.** | **Abtrags- und Akkumulationsformen** |
| 2.1.1 | Flächenhafte Abtragsformen |
| 2.1.2 | Lineare Abtragsformen |
| 2.1.3 | Flächenhaft lineare Abtragsformen |
| 2.1.4.1 | Anzahl linearer Erosionsformen |
| 2.1.4.2 | Mittlere Tiefe der Formen (in cm) |
| 2.1.4.3 | Mittlere Breite der Formen (in cm) |
| 2.1.4.4 | Querschnittsform der Formen |
| 2.1.4.5 | Mittlere Länge der Formen (in m) |
| 2.1.4.6 | Mittleres Abtragsvolumen der Einzelform (in m³) |
| 2.1.4.7 | Ausgetragenes Bodenvolumen aller Formen (in m³) |
| 2.1.5.2 | Lage der Erosionsformen innerhalb der betroffenen Fläche |
| 2.1.5.3 | Größe der vom Bodenabtrag betroffenen Fläche (in m²) |
| 2.1.5.4 | Erosion in Mulden und Tiefenlinien |
| 2.2.1 | Kleinflächige Akkumulation (Fläche < 20 m²) |
| 2.2.2 | Großflächige Akkumulation (Fläche ≥ 20 m²) |
| 2.2.7.4 | Flächengröße der Akkumulation |
| **3.** | **Übertritte von erodiertem Boden, flächenexterne Wirkungen („off-site")** |
| 3.1 | Übertrittsstellen |
| **4.** | **Oberflächenwasser, Fremdwasserzufluss und Wasseraustritt** |
| 4.1 | Fremdwasserzufluss (allgemein) |
| 4.2 | Wasseraustritt auf der von Bodenerosion betroffenen Fläche |
| **5.** | **Relief, Reliefmerkmale** |
| 5.2 | Hangneigung im Bereich der Abtragsform |

### Appendix C: Additional Mapping Parameters

Additional parameters for overview and detailed erosion mapping.

| Lfd. Nr. | Attribut / Kartierobjekt |
|---------:|--------------------------|
| **0.** | **Allgemeine Daten zur Kartierung** |
| 0.8.3 | Gemeinde / Gemarkung, in der die Kartierung stattfindet |
| 0.8.4 | Auftraggeber der Kartierung |
| 0.8.5 | Schadensmeldung durch Dritte |
| 0.8.6 | Bezug zu anderen Schadenskartierungen und Dokumentationsblättern |
| **1.** | **Flächennutzung, Bodenbearbeitung und Bewirtschaftung** |
| 1.3 | Bearbeitungszustand der Ackerfläche |
| 1.4.1 | Bearbeitungsrichtung Hauptfeld |
| 1.4.2 | Bearbeitungsrichtung Vorgewende |
| 1.5.1 | Verschlämmung der Bodenoberfläche |
| 1.5.2 | Steinbedeckung (in %) |
| 1.5.4 | Phänologisches Entwicklungsstadium der Kulturpflanze |
| 1.6 | Ackerbaulicher Erosionsschutz |
| **2.** | **Abtrags- und Akkumulationsformen** |
| 2.1.5.1 | Wiederholter Abtrag an derselben Geländeposition |
| 2.1.6 | Bioindikatoren für Erosionsprozesse |
| 2.2.7.1 | Wiederholte Akkumulation an derselben Geländeposition |
| 2.2.7.2 | Akkumuliertes / deponiertes Bodenvolumen |
| 2.2.7.3 | Schäden an Kulturpflanzen durch Akkumulation |
| **3.** | **Übertritte von erodiertem Boden, flächenexterne Wirkungen („off-site")** |
| 3.2 | Stärke des Off-Site-Schadens |
| **4.** | **Oberflächenwasser, Fremdwasserzufluss und Wasseraustritt** |
| 4.3 | Sonstige wasserhaushaltliche Merkmale |
| **5.** | **Relief, Reliefmerkmale** |
| 5.1 | Reliefgruppentyp |
| 5.3.1 | Erosionswirksame Hanglänge bis zum Beginn der Akkumulation |
| 5.3.2 | Hanglänge bis zum Beginn der Erosionsform / des Erosionssystems |
| 5.4 | Wölbungsform im Bereich der Erosionsform |
| 5.5 | Lage des Erosionssystems im Gelände |
| **6.** | Sonstiger Erosionsschutz |
| 6.1 – 6.6 | z. B. Gehölzflächen, Hecken, Wälle usw. |

---

*© 2024-2026 BAW Research | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)*
