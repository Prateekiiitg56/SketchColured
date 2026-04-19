// ─── DOM Elements ───
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const browseLink = document.getElementById('browse-link');
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
const menuToggle = document.getElementById('menu-toggle');

// ─── State ───
let uploadedFile = null;
let sketchDataURL = null;

// ─── Mobile Menu ───
if (menuToggle) {
    menuToggle.addEventListener('click', () => {
        const navLinks = document.querySelector('.nav-links');
        navLinks.classList.toggle('nav-open');
    });
}

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

        resultColor.src = colorURL;
        resultSketch.src = sketchDataURL;
        downloadBtn.href = colorURL;

        resultSection.classList.remove('hidden');
        previewState.classList.add('hidden');

        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        setSlider(50);

    } catch (err) {
        alert('Colorization failed: ' + err.message);
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
function loadGallery() {
    // Show latest epochs first (most impressive results), then milestones
    const epochs = [
        // Latest training results
        399, 398, 397, 396, 395, 394, 393, 392, 391, 390,
        // Milestone epochs showing training progression
        194, 190, 185, 180, 175, 171
    ];

    epochs.forEach((ep) => {
        const item = document.createElement('div');
        item.className = 'sample-item';
        item.innerHTML = `
      <img src="/static/samples/y_gen_${ep}.png" alt="Epoch ${ep} output" loading="lazy"
           onerror="this.closest('.sample-item').remove()" />
      <div class="epoch-label">Epoch ${ep}</div>
    `;
        sampleGrid.appendChild(item);
    });
}

loadGallery();

// ─── Smooth Scroll for Nav Links ───
document.querySelectorAll('.nav-item[href^="#"]').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const target = document.querySelector(link.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        // Close mobile menu if open
        const navLinks = document.querySelector('.nav-links');
        navLinks.classList.remove('nav-open');
    });
});

// ─── Scroll Reveal ───
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('revealed');
            observer.unobserve(entry.target);
        }
    });
}, { threshold: 0.1 });

document.querySelectorAll('.section-label, .section-title-xl, .how-card, .sample-item').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
    observer.observe(el);
});

// Add revealed class styles
const style = document.createElement('style');
style.textContent = '.revealed { opacity: 1 !important; transform: translateY(0) !important; }';
document.head.appendChild(style);
