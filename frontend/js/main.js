(function () {
  const messagesEl = document.getElementById("messages");
  const messagesEmptyEl = document.getElementById("messages-empty");
  const responsePanelEl = document.getElementById("response-panel");
  const responseContentEl = document.getElementById("response-content");
  const userInputEl = document.getElementById("user-input");
  const imageInputEl = document.getElementById("image-input");
  const attachBtnEl = document.getElementById("attach-btn");
  const sendBtnEl = document.getElementById("send-btn");
  const imagePreviewEl = document.getElementById("image-preview");
  const previewImgEl = document.getElementById("preview-img");
  const imagePathLabelEl = document.getElementById("image-path-label");
  const removeImageBtnEl = document.getElementById("remove-image-btn");

  let selectedImagePath = null;
  let previewObjectUrl = null;

  attachBtnEl.addEventListener("click", () => imageInputEl.click());

  imageInputEl.addEventListener("change", () => {
    const file = imageInputEl.files[0];
    if (!file) return;

    clearImagePreview();

    selectedImagePath = `data/images/${file.name}`;
    previewObjectUrl = URL.createObjectURL(file);

    previewImgEl.src = previewObjectUrl;
    imagePathLabelEl.textContent = selectedImagePath;
    imagePreviewEl.classList.remove("hidden");
  });

  removeImageBtnEl.addEventListener("click", () => {
    clearImagePreview();
    imageInputEl.value = "";
  });

  sendBtnEl.addEventListener("click", handleSend);

  userInputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  });

  async function handleSend() {
    const message = userInputEl.value.trim();
    const imagePath = selectedImagePath;

    if (!message && !imagePath) return;

    setLoading(true);
    hideEmptyState();
    const previewForMessage = previewObjectUrl;
    appendMessage("user", message, imagePath, previewForMessage);
    resetInputImage();

    showResponse("Esperando respuesta del servidor...", true);
    debugLog("Sending message...", { message, imagePath });

    try {
      const data = await sendChatMessage({ message, imagePath });
      debugLog("Got response from server:", data);

      appendMessage("assistant", data.response, data.imagePath);
      showResponse(data.response, false, false, data.imagePath);
    } catch (error) {
      debugLog("Error:", error);
      const errorMsg =
        error.message ||
        "No se pudo conectar con el servidor. Verifica que el endpoint esté activo.";
      showResponse(errorMsg, false, true);
      
      // Log error details for debugging
      console.error("Chat error details:", {
        errorMessage: error.message,
        errorStack: error.stack,
        backendUrl: API_CONFIG.baseUrl,
        endpoint: API_CONFIG.endpoints.chat
      });
    } finally {
      userInputEl.value = "";
      setLoading(false);
      userInputEl.focus();
    }
  }

  function appendMessage(role, text, imagePath, imagePreviewUrl) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;

    const roleLabel = document.createElement("span");
    roleLabel.className = "message-role";
    roleLabel.textContent = role === "user" ? "Tú" : "Asistente";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    if (text) {
      bubble.appendChild(document.createTextNode(text));
    }

    if (imagePath) {
      const pathSpan = document.createElement("span");
      pathSpan.className = "image-path";
      pathSpan.textContent = imagePath;
      bubble.appendChild(pathSpan);

      const imgSrc =
        role === "user" && imagePreviewUrl
          ? imagePreviewUrl
          : resolveImageUrl(imagePath);

      if (imgSrc) {
        const alt = role === "user" ? "Imagen adjunta" : "Imagen de la respuesta";
        bubble.appendChild(createMessageImage(imgSrc, alt));
      }
    }

    wrapper.appendChild(roleLabel);
    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function createMessageImage(src, alt) {
    const img = document.createElement("img");
    img.src = src;
    img.alt = alt;
    img.loading = "lazy";
    return img;
  }

  function showResponse(text, isLoading = false, isError = false, imagePath = null) {
    responsePanelEl.classList.remove("hidden");
    responseContentEl.replaceChildren();
    responseContentEl.classList.toggle("loading", isLoading);
    responseContentEl.classList.toggle("error", isError);

    if (text) {
      responseContentEl.appendChild(document.createTextNode(text));
    }

    if (imagePath && !isLoading && !isError) {
      const pathSpan = document.createElement("span");
      pathSpan.className = "image-path";
      pathSpan.textContent = imagePath;
      responseContentEl.appendChild(pathSpan);
      responseContentEl.appendChild(
        createMessageImage(resolveImageUrl(imagePath), "Imagen de la respuesta")
      );
    }
  }

  function hideEmptyState() {
    messagesEmptyEl.classList.add("hidden");
  }

  function setLoading(loading) {
    sendBtnEl.disabled = loading;
    attachBtnEl.disabled = loading;
    userInputEl.disabled = loading;
  }

  function clearImagePreview() {
    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
    }
    resetInputImage();
  }

  function resetInputImage() {
    previewObjectUrl = null;
    selectedImagePath = null;
    previewImgEl.src = "";
    imagePathLabelEl.textContent = "";
    imagePreviewEl.classList.add("hidden");
    imageInputEl.value = "";
  }
})();
