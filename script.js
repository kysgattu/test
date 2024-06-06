
const confirmBtn = document.getElementById('confirmBtn');
const cancelBtn = document.getElementById('cancelBtn');

confirmBtn.addEventListener('click', () => {
  fetch('/confirm', { method: 'post' }) // Send confirmation response to script
    .then(() => window.close()); // Close the web app window
});

cancelBtn.addEventListener('click', () => {
  fetch('/cancel', { method: 'post' }) // (Optional) Send cancel response to script (if needed)
  .then(() => window.close()); // Close the web app window
});
