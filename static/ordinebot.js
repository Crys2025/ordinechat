console.log("OrdineBot JS loaded");

let openedAutomatically = false;

function toggleChat() {
    const chat = document.getElementById('ai-chat-box');
    if (!chat) {
        console.error("Chat box not found!");
        return;
    }
    chat.style.display = (chat.style.display === 'block') ? 'none' : 'block';
}

function scrollMessages() {
    const box = document.getElementById('ai-chat-messages');
    if (box) box.scrollTop = box.scrollHeight;
}

function addUserMessage(msg) {
    const box = document.getElementById('ai-chat-messages');
    const typing = document.getElementById('typing-indicator');
    if (!box) return;
    if (typing) typing.style.display = "none";

    box.innerHTML += `<div class="user-msg">${msg}</div>`;
    scrollMessages();
}

function addBotMessage(msg) {
    const box = document.getElementById('ai-chat-messages');
    const typing = document.getElementById('typing-indicator');
    if (!box) return;
    if (typing) typing.style.display = "none";

    box.innerHTML += `<div class="bot-msg">${msg}</div>`;
    scrollMessages();
}

function showTyping() {
    const typing = document.getElementById('typing-indicator');
    if (typing) typing.style.display = "block";
    scrollMessages();
}

/*** AUTO-DESCHIDERE după 3 secunde ***/
setTimeout(() => {
    if (!openedAutomatically && !sessionStorage.getItem("botOpened")) {
        openedAutomatically = true;
        sessionStorage.setItem("botOpened", "yes");
        toggleChat();
        setTimeout(() => {
            addBotMessage("Bună! 👋 Sunt OrdineBot. Te pot ajuta să găsești ceva pe site?");
        }, 500);
    }
}, 3000);

/*** TRIMITERE MESAJ ***/
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
            body: JSON.stringify({ query: msg })
        });

        const data = await response.json();
        addBotMessage(data.answer);
    } catch (error) {
        console.error(error);
        addBotMessage("Îmi pare rău 😔 Nu mă pot conecta la server chiar acum.");
    }
}

/*** ASOCIERE CLICK ***/
document.addEventListener("DOMContentLoaded", () => {
    const bubble = document.getElementById("ai-bot-bubble");
    const input = document.getElementById("ai-chat-input");

    if (bubble) bubble.onclick = toggleChat;

    if (input) {
        input.addEventListener("keydown", function (event) {
            if (event.key === "Enter") sendMessage();
        });
    }
});
