const CHANNEL_NAME = "collectorvision-monitor";
const DEFAULT_HIDE_MS = 8000;

const card = document.getElementById("overlay-card");
const nameEl = document.getElementById("card-name");
const metaEl = document.getElementById("card-meta");
let hideTimer = null;

function cardLabel(event) {
  return event?.card?.name
    || event?.card?.cardId
    || "Unknown card";
}

function cardMeta(event) {
  const score = Number(event?.card?.score);
  const setCode = event?.card?.setCode ? String(event.card.setCode).toUpperCase() : "";
  const parts = [];
  if (setCode) parts.push(setCode);
  if (Number.isFinite(score)) parts.push(`score ${score.toFixed(3)}`);
  return parts.join(" · ") || "CollectorVision";
}

function showEvent(event) {
  nameEl.textContent = cardLabel(event);
  metaEl.textContent = cardMeta(event);
  card.hidden = false;
  requestAnimationFrame(() => card.classList.add("is-visible"));

  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => {
    card.classList.remove("is-visible");
  }, DEFAULT_HIDE_MS);
}

const channel = new BroadcastChannel(CHANNEL_NAME);
channel.addEventListener("message", (event) => {
  if (event.data?.type === "collectorvision.card") {
    showEvent(event.data);
  }
});
