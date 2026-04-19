// ─── DOM Elements ───
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

// ─── State ───
let uploadedFile = null;
let sketchDataURL = null;

// ─── Upload Interactions ───
browseLink.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});
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
        showToast('Please upload an image file (PNG, JPG, WEBP).', 'error');
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

// ─── Toggle Logic ───
colorTransferToggle.addEventListener('change', () => {
    if (colorTransferToggle.checked) {
        referenceUploadGroup.classList.remove('hidden');
    } else {
        referenceUploadGroup.classList.add('hidden');
        referenceInput.value = '';
    }
});

// ─── Reset / Try Again ───
function resetUpload() {
    uploadedFile = null;
    sketchDataURL = null;
    fileInput.value = '';
    previewState.classList.add('hidden');
    resultSection.classList.add('hidden');
    dropZone.classList.remove('hidden');
    colorTransferToggle.checked = false;
    referenceUploadGroup.classList.add('hidden');
    referenceInput.value = '';
}

resetBtn.addEventListener('click', resetUpload);
tryAgainBtn.addEventListener('click', resetUpload);

// ─── API Call ───
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

        // Hide preview state
        previewState.classList.add('hidden');

        // Smooth scroll to result
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Reset slider to center
        setSlider(50);

        showToast('Colorization complete! 🎨', 'success');

    } catch (err) {
        showToast('Colorization failed: ' + err.message, 'error');
    } finally {
        loadingOverlay.classList.add('hidden');
        colorizeBtn.disabled = false;
    }
});

// ─── Before/After Slider ───
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

// ─── Training Gallery ───
// Show the latest epochs from the evaluation folder
// Check both /static/samples and latestOutput evaluation  
function loadGallery() {
    // Try the latest training output first (epochs 390-399), fallback to static samples
    const latestEpochs = [];
    for (let i = 395; i <= 399; i++) latestEpochs.push(i);
    for (let i = 390; i <= 394; i++) latestEpochs.push(i);

    // Also add older milestone epochs from static samples
    const milestoneEpochs = [171, 175, 180, 185, 190, 194];

    // Load latest training epochs from evaluation folder
    latestEpochs.forEach((ep) => {
        const item = document.createElement('div');
        item.className = 'sample-item';
        item.innerHTML = `
      <img src="/static/samples/y_gen_${ep}.png" alt="Epoch ${ep}" loading="lazy"
           onerror="this.closest('.sample-item').remove()" />
      <div class="epoch-label">Epoch ${ep}</div>
    `;
        sampleGrid.appendChild(item);
    });

    // Load milestone epochs
    milestoneEpochs.forEach((ep) => {
        const item = document.createElement('div');
        item.className = 'sample-item';
        item.innerHTML = `
      <img src="/static/samples/y_gen_${ep}.png" alt="Epoch ${ep}" loading="lazy"
           onerror="this.closest('.sample-item').remove()" />
      <div class="epoch-label">Epoch ${ep}</div>
    `;
        sampleGrid.appendChild(item);
    });
}

loadGallery();

// ─── Toast Notification ───
function showToast(message, type = 'info') {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    Object.assign(toast.style, {
        position: 'fixed',
        bottom: '24px',
        right: '24px',
        padding: '14px 24px',
        borderRadius: '12px',
        fontSize: '0.88rem',
        fontWeight: '600',
        fontFamily: "'Outfit', sans-serif",
        zIndex: '9999',
        animation: 'fadeInUp 0.4s ease-out',
        maxWidth: '380px',
        backdropFilter: 'blur(12px)',
    });

    if (type === 'success') {
        toast.style.background = 'rgba(52, 211, 153, 0.15)';
        toast.style.color = '#34d399';
        toast.style.border = '1px solid rgba(52, 211, 153, 0.3)';
    } else if (type === 'error') {
        toast.style.background = 'rgba(248, 113, 113, 0.15)';
        toast.style.color = '#f87171';
        toast.style.border = '1px solid rgba(248, 113, 113, 0.3)';
    } else {
        toast.style.background = 'rgba(167, 139, 250, 0.15)';
        toast.style.color = '#a78bfa';
        toast.style.border = '1px solid rgba(167, 139, 250, 0.3)';
    }

    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'fadeInUp 0.3s ease-in reverse forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}
