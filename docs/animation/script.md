# Pipeline Explained — Animation Cut Sheet

**Runtime target:** ~90 seconds  
**Style:** 3Blue1Brown — dark background, smooth transitions, mathematical precision  
**Narration:** Text captions only (no voice-over required to render)

---

## Scene 1 — Title (5s)

**Visuals:**
- Fade in: "How does a computer recognize a trading card?" in large white text
- Subtitle fades in: "CollectorVision — a three-step pipeline"
- Background: dark (#1C1C2E), subtle grid

**Cut text:** *"You hold up a card. A camera sees pixels. How does software get from there to a name?"*

---

## Scene 2 — The Problem (8s)

**Visuals:**
- A photo of the Scrying Glass card appears (skewed, real-world photo)
- A question mark pulses next to it
- Three small boxes appear below labeled: "Detect", "Dewarp", "Identify"
- Arrow sweeps left-to-right through all three

**Cut text:** *"The pipeline has three steps: find the card corners, flatten the perspective, then match the image to a catalog of 108,000 reference cards."*

---

## Scene 3 — Step 1: Corner Detection (20s)

**Visuals:**
1. Raw photo fills left side of frame
2. Title "Step 1: Corner Detection" appears top-right
3. Four colored blobs fade in at each corner (TL=red, TR=cyan, BR=yellow, BL=magenta)
   - Blobs are SimCC heatmap stand-ins — soft Gaussian circles
   - Label: "Neural network predicts where each corner is"
4. Four dots snap to their peak positions with coordinates shown
5. Lines connect the four dots to form a quadrilateral overlay
6. "Confidence: sharpness = 0.045" appears as a small readout
   - Brief explanation: "High sharpness = sharp prediction peaks = card is present"

**Cut text:**
- *"A neural network (Cornelius) looks at the full image and outputs the location of all four card corners."*
- *"Internally it uses SimCC: separate probability distributions for X and Y of each corner."*
- *"When the distributions have sharp peaks, the card is there. Flat = no card."*

---

## Scene 4 — Step 2: Dewarping (15s)

**Visuals:**
1. The skewed quadrilateral from Scene 3 animates — corners labeled TL/TR/BR/BL
2. A perspective transform morphs the quad into Milo's square input crop (448 × 448 px target)
   - The transform should look like the card is being "pressed flat"
   - Use a smooth warp animation: each corner traces an arc to its destination
3. The dewarped card image appears, clean and rectangular
4. Small text: "448 × 448 px — exactly what the embedder sees"

**Cut text:**
- *"Using the four corners, we apply a perspective transform — the same math used in map projections."*
- *"The result is a clean, flat card image, always the same size."*

---

## Scene 5 — Step 3: The Embedding (25s)

**Visuals:**
1. Dewarped card feeds into a funnel diagram (MobileViT-XXS)
   - Input: 448 × 448 px box labeled "Card image"
   - Middle: abstract layered rectangles shrinking (the neural net)
   - Output: "128 numbers"
2. The 128 numbers visualized as a horizontal strip of colored cells (the "barcode")
   - Cells are colored on a blue-to-red heatmap
   - Strip is labeled "Embedding vector — 128 dimensions, L2-normalized"
3. Key insight text: "Similar cards → similar vectors"
   - Show two reference card barcodes (from catalog) appearing below
   - One matches closely (similar color pattern), one doesn't
4. Zoom to the barcode comparison — show how the query strip aligns with the matching entry

**Cut text:**
- *"The neural net (Milo) converts the flat card image into 128 numbers — a compact fingerprint."*
- *"Every card in our 108,000-card catalog has been pre-embedded the same way."*
- *"The same card always produces a similar fingerprint, regardless of lighting or angle."*

---

## Scene 6 — Nearest-Neighbor Search (15s)

**Visuals:**
1. Query barcode on left, arrow pointing right
2. Catalog appears as a stack of cards (compressed visual — dots representing 108k cards)
3. The search: query barcode slides along the catalog stack as scores appear
   - 5 candidate rows with their barcodes and cosine similarity scores
   - Best match highlighted green: score = 0.846
   - Others fade slightly
4. Best match card thumbnail revealed: "Scrying Glass"
   - "Match found in ~8 ms"

**Cut text:**
- *"We compare the query fingerprint against all 108,000 catalog embeddings using cosine similarity."*
- *"The closest match wins. This runs in milliseconds on a CPU."*

---

## Scene 7 — Result (7s)

**Visuals:**
1. Original photo (from Scene 2) on left
2. Large arrow "→"
3. Card name, set, and artist appear on right: "Scrying Glass · Urza's Destiny · rk post"
4. Score badge: "0.846"
5. Final fade to: "CollectorVision — collector_vision on PyPI"

**Cut text:** *"That's it. A real-world photo, identified in under a second."*

---

## Technical Notes

- All scene transitions: `FadeTransform` or slide wipes, 0.5s
- Color palette: background `#1C1C2E`, text white, accent `#58C4DD` (3b1b blue), highlight `#00E678`
- Corner heatmap colors: TL=`#FF4444`, TR=`#44FFFF`, BR=`#FFFF44`, BL=`#FF44FF`
- Embedding barcode: `values → np.clip((values+1)/2, 0, 1) → blue_to_red colormap`
- Font: Noto Sans or system default (manim uses Cairo text rendering)
