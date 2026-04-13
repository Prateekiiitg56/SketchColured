// DOM elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const browseLink = document.getElementById('browse-link');
const uploadSection = document.getElementById('upload-section');
const previewState = document.getElementById('preview-state');
const sketchThumb = document.getElementById('sketch-thumb');
const colorizeBtn = document.getElementById('colorize-btn');
const resetBtn = document.getElementById('reset-btn');

const resultSection = document.getElementById('result-section');
const resultColor = document.getElementById('result-color');
const resultSketch = document.getElementById('result-sketch');
const downloadBtn = document.getElementById('download-btn');
const tryAgainBtn = document.getElementById('try-again-btn');

const colorTransferToggle = document.getElementById('color-transfer-toggle');
const referenceUploadGroup = document.getElementById('reference-upload-group');
const referenceInput = document.getElementById('reference-input');

const loadingOverlay = document.getElementById('loading-overlay');
const sampleGrid = document.getElementById('sample-grid');
const compareClip = document.getElementById('compare-clip');
const compareHandle = document.getElementById('compare-handle');
const compareWrap = document.getElementById('compare-container');

// State
let uploadedFile = null;
let sketchDataURL = null;

// --- Upload Interaction ---
browseLink.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (PNG, JPG, WEBP).');
        return;
    }
    uploadedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        sketchDataURL = e.target.result;
        sketchThumb.src = sketchDataURL;
        dropZone.classList.add('hidden');
        previewState.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

// --- Toggle Logic ---
colorTransferToggle.addEventListener('change', () => {
    if (colorTransferToggle.checked) {
        referenceUploadGroup.classList.remove('hidden');
    } else {
        referenceUploadGroup.classList.add('hidden');
        referenceInput.value = '';
    }
});

// --- Reset / Try Again ---
function resetUpload() {
    uploadedFile = null;
    sketchDataURL = null;
    fileInput.value = '';
    previewState.classList.add('hidden');
    resultSection.classList.add('hidden');
    dropZone.classList.remove('hidden');
}

resetBtn.addEventListener('click', resetUpload);
tryAgainBtn.addEventListener('click', resetUpload);


// --- API Call ---
colorizeBtn.addEventListener('click', async () => {
    if (!uploadedFile) return;

    loadingOverlay.classList.remove('hidden');
    colorizeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', uploadedFile);
    const useColorTransfer = colorTransferToggle.checked;
    formData.append('use_color_transfer', useColorTransfer);

    if (useColorTransfer && referenceInput.files[0]) {
        formData.append('reference_file', referenceInput.files[0]);
    }

    try {
        const resp = await fetch('/colorize', { method: 'POST', body: formData });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(err.detail || `Server error ${resp.status}`);
        }

        const blob = await resp.blob();
        const colorURL = URL.createObjectURL(blob);

        // Set slider images
        resultColor.src = colorURL;
        resultSketch.src = sketchDataURL;
        downloadBtn.href = colorURL;

        // Show result
        resultSection.classList.remove('hidden');

        // Hide preview state entirely once colorized
        previewState.classList.add('hidden');

        // Smooth scroll down to result
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Reset slider to center
        setSlider(50);

    } catch (err) {
        alert('Model inference failed: ' + err.message);
    } finally {
        loadingOverlay.classList.add('hidden');
        colorizeBtn.disabled = false;
    }
});


// --- Slider Logic ---
function setSlider(pct) {
    compareClip.style.width = pct + '%';
    compareHandle.style.left = pct + '%';
}

let isDragging = false;

compareWrap.addEventListener('mousedown', startDrag);
compareWrap.addEventListener('touchstart', startDrag, { passive: true });

window.addEventListener('mousemove', onDrag);
window.addEventListener('touchmove', onDrag, { passive: false });
window.addEventListener('mouseup', stopDrag);
window.addEventListener('touchend', stopDrag);

function startDrag(e) { isDragging = true; onDrag(e); }
function stopDrag() { isDragging = false; }

function onDrag(e) {
    if (!isDragging) return;
    if (e.cancelable) e.preventDefault();

    const rect = compareWrap.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;

    let pct = ((clientX - rect.left) / rect.width) * 100;
    pct = Math.min(100, Math.max(0, pct));
    setSlider(pct);
}

// --- Training Gallery ---
const epochs = [187, 188, 189, 190, 191, 192, 193, 194];

epochs.forEach((ep) => {
    const item = document.createElement('div');
    item.className = 'sample-item';
    item.innerHTML = `
    <img src="/static/samples/y_gen_${ep}.png" alt="Epoch ${ep}" loading="lazy" onerror="this.closest('.sample-item').remove()" />
    <div class="epoch-label">Epoch ${ep}</div>
  `;
    sampleGrid.appendChild(item);
});
