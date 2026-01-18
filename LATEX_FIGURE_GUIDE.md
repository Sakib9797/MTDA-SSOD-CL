# YOLOv11s Architecture Figure - LaTeX Integration Guide

## Files Generated
- `yolov11s_architecture.pdf` ✓ **USE THIS** (Best for LaTeX/Overleaf)
- `yolov11s_architecture.png` (Backup/presentations)
- `yolov11s_architecture.eps` (Alternative LaTeX format)

---

## STEP 1: Upload Figure to Overleaf

1. In your Overleaf project, locate the `figures/` or `images/` folder
   - If it doesn't exist, create a new folder called `figures`
2. Click **Upload** (top left)
3. Select `yolov11s_architecture.pdf` from your computer
4. The file should now appear in your project file tree

---

## STEP 2: LaTeX Code Options

### Option A: Single Column Figure (RECOMMENDED)
For standard single-column IEEE papers:

```latex
\begin{figure}[!htbp]
\centering
\includegraphics[width=\columnwidth]{figures/yolov11s_architecture.pdf}
\caption{Detailed YOLOv11s model architecture showing backbone feature extraction, neck fusion, and detection head layers. The model consists of 9.4M parameters and processes 640×640 input images through convolutional layers, C2f blocks, SPPF, and multi-scale detection heads.}
\label{fig:yolov11s_architecture}
\end{figure}
```

### Option B: Full-Width Figure (Two-Column Papers)
If you want the figure to span both columns:

```latex
\begin{figure*}[!htbp]
\centering
\includegraphics[width=0.5\textwidth]{figures/yolov11s_architecture.pdf}
\caption{Detailed YOLOv11s model architecture showing backbone feature extraction, neck fusion, and detection head layers. The model consists of 9.4M parameters and processes 640×640 input images through convolutional layers, C2f blocks, SPPF, and multi-scale detection heads.}
\label{fig:yolov11s_architecture}
\end{figure*}
```

### Option C: Rotated/Landscape Figure
For portrait-oriented diagrams that work better rotated:

```latex
\begin{figure}[!htbp]
\centering
\includegraphics[width=\columnwidth]{figures/yolov11s_architecture.pdf}
\caption{Detailed YOLOv11s model architecture showing backbone feature extraction, neck fusion, and detection head layers.}
\label{fig:yolov11s_architecture}
\end{figure}
```

---

## STEP 3: Required LaTeX Packages

Add these to your preamble if not already present:

```latex
\usepackage{graphicx}  % For \includegraphics
\usepackage{float}     % For [H] placement option (optional)
```

---

## STEP 4: Referencing the Figure in Text

In your paper text, reference the figure using:

```latex
As shown in Fig.~\ref{fig:yolov11s_architecture}, the YOLOv11s architecture consists of three main components...
```

Or:

```latex
The backbone network (Figure~\ref{fig:yolov11s_architecture}) extracts features at multiple scales...
```

---

## STEP 5: Placement Options Explained

The `[!htbp]` options control where LaTeX places the figure:
- `h` = here (preferred position)
- `t` = top of page
- `b` = bottom of page
- `p` = separate page
- `!` = override LaTeX's internal parameters

**Common alternatives:**
- `[H]` = Place exactly HERE (requires `\usepackage{float}`)
- `[!t]` = Force top of page
- `[!b]` = Force bottom of page

---

## Complete Example Section

```latex
\section{Model Architecture}
\label{sec:architecture}

Our object detection framework employs YOLOv11s as the base architecture. 
Figure~\ref{fig:yolov11s_architecture} illustrates the complete network 
structure, which comprises three main components: the backbone for feature 
extraction, the neck for feature fusion, and the detection head for 
final predictions.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\columnwidth]{figures/yolov11s_architecture.pdf}
\caption{Detailed YOLOv11s model architecture showing backbone feature 
extraction, neck fusion, and detection head layers. The model consists of 
9.4M parameters and processes 640×640 input images through convolutional 
layers, C2f blocks, SPPF, and multi-scale detection heads.}
\label{fig:yolov11s_architecture}
\end{figure}

The backbone network consists of multiple convolutional layers and C2f 
blocks, which progressively downsample the input image while extracting 
hierarchical features. As shown in Fig.~\ref{fig:yolov11s_architecture}, 
the network produces feature maps at three different scales (P3/8, P4/16, 
and P5/32), enabling detection of objects at various sizes.

The neck utilizes a Feature Pyramid Network (FPN) structure with upsampling 
and concatenation operations to merge features from different scales...
```

---

## Troubleshooting

### Issue 1: "File not found" error
**Solution:** Check the path in `\includegraphics{}`:
- If figure is in root: `{yolov11s_architecture.pdf}`
- If in folder: `{figures/yolov11s_architecture.pdf}`

### Issue 2: Figure appears too large/small
**Solution:** Adjust the width parameter:
```latex
\includegraphics[width=0.9\columnwidth]{...}  % 90% of column
\includegraphics[width=0.8\columnwidth]{...}  % 80% of column
\includegraphics[height=0.7\textheight]{...}  % By height instead
```

### Issue 3: Figure on wrong page
**Solution:** Try different placement options:
```latex
\begin{figure}[!t]  % Force top of page
\begin{figure}[!h]  % Try to place here
\begin{figure}[H]   % Exact position (needs \usepackage{float})
```

### Issue 4: Figure too blurry
**Solution:** Make sure you're using the PDF version:
- ✓ `yolov11s_architecture.pdf` (vector, scales perfectly)
- ✗ `yolov11s_architecture.png` (raster, may blur)

---

## Advanced: Side-by-Side with Another Figure

```latex
\begin{figure*}[!htbp]
\centering
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/yolov11s_architecture.pdf}
    \caption{YOLOv11s architecture}
    \label{fig:architecture}
\end{minipage}
\hfill
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/training_curve.pdf}
    \caption{Training progression}
    \label{fig:training}
\end{minipage}
\end{figure*}
```

---

## Quick Reference: Size Adjustments

| Code | Result |
|------|--------|
| `width=\columnwidth` | Full column width (recommended) |
| `width=0.9\columnwidth` | 90% of column |
| `width=\textwidth` | Full text width (for figure*) |
| `width=0.5\textwidth` | Half text width |
| `scale=0.8` | 80% of original size |
| `height=6cm` | Specific height |

---

## File Locations in Your Project

```
your-overleaf-project/
├── main.tex
├── sections/
│   ├── introduction.tex
│   ├── methodology.tex    ← Add figure here
│   └── results.tex
└── figures/                ← Upload PDF here
    └── yolov11s_architecture.pdf
```

---

## Summary Checklist

- [ ] Generate figure using `generate_yolov11s_architecture.py`
- [ ] Upload `yolov11s_architecture.pdf` to Overleaf
- [ ] Add `\usepackage{graphicx}` to preamble
- [ ] Insert figure code in appropriate section
- [ ] Compile and check rendering
- [ ] Reference figure in text using `\ref{fig:yolov11s_architecture}`
- [ ] Adjust width/placement if needed

---

## Contact & Support

If the figure doesn't appear correctly:
1. Check Overleaf compile logs for errors
2. Verify file path matches your folder structure
3. Ensure figure is uploaded successfully
4. Try recompiling the document (Ctrl+S or Recompile button)

**The figure is optimized at 600 DPI and will be perfectly clear without any zooming needed!**
