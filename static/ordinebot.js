console.log("OrdineBot JS loaded");

let openedAutomatically = false;
let pgConversation = [];

/* ======================================================
   FUNCTIE LINKIFY - transforma linkurile in <a>
====================================================== */
function linkify(text) {
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    return text.replace(urlRegex, url => {
        return `<a href="${url}" class="ai-link">${url}</a>`;
    });
}

/* ======================================================
   DESCHIDE / INCHIDE FEREASTRA - VARIANTA FINALA (.open)
====================================================== */
function toggleChat() {
    const chat = document.getElementById('ai-chat-box');
    if (!chat) return;

    chat.classList.toggle("open");
    scrollMessages();
}

// expunem global pentru HTML
window.toggleChat = toggleChat;

/* ======================================================
   SCROLL
====================================================== */
function scrollMessages() {
    const box = document.getElementById('ai-chat-messages');
    if (!box) return;
    box.scrollTop = box.scrollHeight;
}

/* ======================================================
   SALVARE CONVERSATIE
====================================================== */
function saveChat() {
    const box = document.getElementById("ai-chat-messages");
    if (!box) return;

    // FIX: numele corect și unic
    sessionStorage.setItem("OrdineBotHistory", box.innerHTML);
}

/* ======================================================
   ADAUGĂ MESAJ USER
====================================================== */
function addUserMessage(msg) {
    const box = document.getElementById('ai-chat-messages');
    if (!box) return;

    box.innerHTML += `<div class="user-msg">${msg}</div>`;
    pgConversation.push({ role: "user", content: msg });

    scrollMessages();
    saveChat();
}

/* ======================================================
   ADAUGĂ MESAJ BOT
====================================================== */
function addBotMessage(msg) {
    const box = document.getElementById('ai-chat-messages');
    if (!box) return;

    box.innerHTML += `<div class="bot-msg">${linkify(msg)}</div>`;
    pgConversation.push({ role: "assistant", content: msg });

    scrollMessages();
    saveChat();
}

/* ======================================================
   ANIMATIA DE TYPING
====================================================== */
function showTyping() {
    const typingBox = document.getElementById("ai-typing");
    typingBox.style.display = "flex";
    scrollMessages();
}

function hideTyping() {
    const typingBox = document.getElementById("ai-typing");
    typingBox.style.display = "none";
}

/* ======================================================
   AUTO DESCHIDERE DUPA 5 SECUNDE LA PRIMA VIZITA
====================================================== */
function autoOpenChat() {
    if (sessionStorage.getItem("GemeniBotAutoOpened")) return;

    setTimeout(() => {
        toggleChat();
        addBotMessage("Bună! Sunt OrdineBot 💗 Cu ce pot să te ajut astăzi?");
        sessionStorage.setItem("GemeniBotAutoOpened", "1");
    }, 5000);
}

/* ======================================================
   TRIMITE MESAJ
====================================================== */
async function sendMessage() {
    const input = document.getElementById('ai-chat-input');
    if (!input) return;

    const msg = input.value.trim();
    if (!msg) return;

    addUserMessage(msg);
    input.value = "";

    showTyping();

    try {
        const response = await fetch("https://ordinechat.onrender.com/ask", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ messages: pgConversation })
        });

        const data = await response.json();
        hideTyping();
        addBotMessage(data.answer);

    } catch (err) {
        hideTyping();
        addBotMessage("❌ Serverul nu răspunde acum. Încearcă din nou.");
    }
}

/* ======================================================
   INITIALIZARE
====================================================== */
document.addEventListener("DOMContentLoaded", () => {
    const messagesBox = document.getElementById("ai-chat-messages");
    const bubble = document.getElementById("ai-bot-bubble");
    const input = document.getElementById("ai-chat-input");
    const sendBtn = document.getElementById("ai-chat-send");

    /* FIX: Restaurăm conversația corect */
    const saved = sessionStorage.getItem("OrdineBotHistory");
    if (saved && messagesBox) {
        messagesBox.innerHTML = saved;

        const nodes = messagesBox.querySelectorAll(".user-msg, .bot-msg");
        nodes.forEach(el => {
            const role = el.classList.contains("user-msg") ? "user" : "assistant";
            const content = el.textContent;
            pgConversation.push({ role, content });
        });

        scrollMessages();
    }

    if (bubble) bubble.onclick = toggleChat;
    if (sendBtn) sendBtn.onclick = sendMessage;

    if (input) {
        input.addEventListener("keydown", ev => {
            if (ev.key === "Enter") sendMessage();
        });
    }

    autoOpenChat();
});







