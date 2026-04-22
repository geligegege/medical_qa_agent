const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const responseContainer = document.getElementById("response");
const messageTemplate = document.getElementById("message-template");

function addMessage(text, role = "bot", isError = false) {
    const messageNode = messageTemplate.content.firstElementChild.cloneNode(true);
    const bubbleNode = messageNode.querySelector(".bubble");

    messageNode.classList.add(`message-${role}`);
    if (isError) {
        messageNode.classList.add("message-error");
    }

    bubbleNode.textContent = text;
    responseContainer.appendChild(messageNode);
    responseContainer.scrollTop = responseContainer.scrollHeight;
    return messageNode;
}

function addTypingMessage() {
    const typingNode = messageTemplate.content.firstElementChild.cloneNode(true);
    typingNode.classList.add("message-bot", "message-system");
    const bubbleNode = typingNode.querySelector(".bubble");
    bubbleNode.innerHTML = "<span class=\"typing-dots\"><span></span><span></span><span></span></span>";
    responseContainer.appendChild(typingNode);
    responseContainer.scrollTop = responseContainer.scrollHeight;
    return typingNode;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();
    if (!question) {
        return;
    }

    questionInput.value = "";
    addMessage(question, "user");

    const sendButton = form.querySelector("button[type='submit']");
    sendButton.disabled = true;
    const typingNode = addTypingMessage();

    try {
        const response = await fetch("/answer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question }),
        });

        if (!response.ok) {
            throw new Error(`Request failed with status ${response.status}`);
        }

        const data = await response.json();
        typingNode.remove();

        const botReply = data.llm_output || data.message || "No response received.";
        addMessage(botReply, "bot");
    } catch (error) {
        console.error(error);
        typingNode.remove();
        addMessage(
            "Sorry, I cannot answer right now. Please try again in a moment.",
            "bot",
            true
        );
    } finally {
        sendButton.disabled = false;
        questionInput.focus();
    }
});
