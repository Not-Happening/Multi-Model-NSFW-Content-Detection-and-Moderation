const API_URL = 'http://localhost:5000/classify';
const classifiedImages = new Set();

async function classifyAndCoverImages() {
  const images = document.querySelectorAll('img');
  
  for (let img of images) {
    // Skip if already classified or no src
    if (classifiedImages.has(img.src) || !img.src || !img.complete) continue;
    
    // Skip data URIs and very small images
    if (img.src.startsWith('data:') || img.width < 50 || img.height < 50) continue;
    
    classifiedImages.add(img.src);
    
    try {
      const result = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageUrl: img.src })
      }).then(r => r.json());
      
      if (result.isNSFW) {
        coverImageWithFact(img, result.fact, result.label);
      }
    } catch (error) {
      console.error('Classification error:', error);
      // Continue with other images even if one fails
    }
  }
}

function coverImageWithFact(imgElement, fact, label) {
  const container = document.createElement('div');
  const width = imgElement.width || 300;
  const height = imgElement.height || 300;
  
  container.style.width = width + 'px';
  container.style.height = height + 'px';
  container.style.backgroundColor = 'black';
  container.style.display = 'flex';
  container.style.alignItems = 'center';
  container.style.justifyContent = 'center';
  container.style.color = 'white';
  container.style.fontSize = '14px';
  container.style.padding = '20px';
  container.style.textAlign = 'center';
  container.style.wordWrap = 'break-word';
  container.style.boxSizing = 'border-box';
  container.style.overflow = 'hidden';
  container.innerHTML = `<div><p><strong>Content Blocked</strong></p><p style="margin-top: 10px; font-size: 12px;">${fact}</p></div>`;
  
  imgElement.replaceWith(container);
}

// Run on initial page load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', classifyAndCoverImages);
} else {
  classifyAndCoverImages();
}

// Monitor for dynamically loaded images
const observer = new MutationObserver(() => {
  classifyAndCoverImages();
});

observer.observe(document.body, { 
  childList: true, 
  subtree: true,
  attributes: false 
});