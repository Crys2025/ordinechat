console.log("OrdineBot JS loaded");

let openedAutomatically = false;

function toggleChat() {
    const chat = document.getElementById('ai-chat-box');
    chat.style.display = (chat.style.display === 'block') ? 'none' : 'block';
}

function scrollMessages() {
    const box = document.getElementById('ai-chat-messages');
    box.scrollTop = box.scrollHeight;
}

function addUserMessage(msg) {
    const box = document.getElementById('ai-chat-messages');
    box.innerHTML += `<div class="user-msg">${msg}</div>`;
    scrollMessages();
}

function addBotMessage(msg) {
    const box = document.getElementById('ai-chat-messages');
    box.innerHTML += `<div class="bot-msg">${msg}</div>`;
    scrollMessages();
}

async function sendMessage() {
    const input = document.getElementById('ai-chat-input');
    const msg = input.value.trim();
    if (!msg) return;

    addUserMessage(msg);
    input.value = "";

    addBotMessage("Scriu răspunsul...");

    try {
        const response = await fetch("https://ordinechat.onrender.com/ask", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ query: msg })
        });

        const data = await response.json();

        document.querySelector(".bot-msg:last-child").remove();
        addBotMessage(data.answer);

    } catch (err) {
        addBotMessage("❌ Serverul nu răspunde. Mai încearcă puțin.");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("ai-bot-bubble").onclick = toggleChat;

    document.getElementById("ai-chat-input").addEventListener("keydown", ev => {
        if (ev.key === "Enter") sendMessage();
    });
});

