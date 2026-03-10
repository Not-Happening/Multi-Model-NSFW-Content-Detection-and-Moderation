const API_URL = 'http://localhost:5050/classify';
const classifiedImages = new WeakSet();
const CONFIDENCE_THRESHOLD = 0.75;

async function classifyImage(img) {
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageUrl: img.src })
    });

    return await response.json();
  } catch (err) {
    console.error("Classification error:", err);
    return null;
  }
}

function createOverlay(img, fact) {
  const wrapper = document.createElement('div');
  wrapper.style.position = 'relative';
  wrapper.style.display = 'inline-block';

  // Preserve size
  wrapper.style.width = img.width + 'px';
  wrapper.style.height = img.height + 'px';

  // Clone image into wrapper
  img.parentNode.insertBefore(wrapper, img);
  wrapper.appendChild(img);

  // Optional blur
  img.style.filter = 'blur(8px)';

  const overlay = document.createElement('div');
  overlay.style.position = 'absolute';
  overlay.style.top = '0';
  overlay.style.left = '0';
  overlay.style.width = '100%';
  overlay.style.height = '100%';
  overlay.style.backgroundColor = 'rgba(0,0,0,0.95)';
  overlay.style.color = 'white';
  overlay.style.display = 'flex';
  overlay.style.flexDirection = 'column';
  overlay.style.justifyContent = 'center';
  overlay.style.alignItems = 'center';
  overlay.style.textAlign = 'center';
  overlay.style.padding = '20px';
  overlay.style.boxSizing = 'border-box';
  overlay.style.fontSize = '14px';
  overlay.style.animation = 'fadeIn 0.3s ease';

  overlay.innerHTML = `
    <div>
      <p style="font-weight:bold; font-size:16px;">Content Blocked</p>
      <p style="margin-top:10px; font-size:13px;">${fact}</p>
    </div>
  `;

  wrapper.appendChild(overlay);
}

async function scanImages() {
  const images = document.querySelectorAll('img');

  for (let img of images) {
    if (!img.src || !img.complete) continue;
    if (img.src.startsWith('data:')) continue;
    if (img.width < 80 || img.height < 80) continue;
    if (classifiedImages.has(img)) continue;

    classifiedImages.add(img);

    const result = await classifyImage(img);
    if (!result) continue;

    if (result.isNSFW && result.confidence > CONFIDENCE_THRESHOLD) {
      createOverlay(img, result.fact);
    }
  }
}

// Initial scan
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', scanImages);
} else {
  scanImages();
}

// Observe dynamic content
const observer = new MutationObserver(() => {
  scanImages();
});

observer.observe(document.body, {
  childList: true,
  subtree: true
});
