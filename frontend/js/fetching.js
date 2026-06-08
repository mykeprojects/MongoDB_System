/**
 * Configuración de endpoints.
 * Ajusta API_BASE y las rutas cuando el backend esté disponible.
 */
const API_CONFIG = {
  baseUrl: "http://localhost:8000",
  endpoints: {
    chat: "/api/chat",
  },
};

/**
 * Convierte una ruta de imagen del servidor en una URL cargable por el navegador.
 *
 * @param {string|null|undefined} imagePath
 * @returns {string|null}
 */
function resolveImageUrl(imagePath) {
  if (!imagePath) return null;

  const normalized = imagePath.replace(/\\/g, "/").trim();
  if (/^https?:\/\//i.test(normalized)) return normalized;

  const withoutLeadingSlash = normalized.replace(/^\//, "");

  if (API_CONFIG.baseUrl) {
    return `${API_CONFIG.baseUrl}/${withoutLeadingSlash}`;
  }

  return `../../${withoutLeadingSlash}`;
}

/**
 * Normaliza la respuesta del servidor a un formato consistente.
 *
 * @param {Record<string, unknown>} data
 * @returns {{ response: string, imagePath: string|null }}
 */
function normalizeChatResponse(data) {
  const response =
    typeof data.response === "string"
      ? data.response
      : typeof data.message === "string"
        ? data.message
        : JSON.stringify(data, null, 2);

  const imagePath =
    typeof data.imagePath === "string" && data.imagePath.trim()
      ? data.imagePath.trim()
      : null;

  return { response, imagePath };
}

/**
 * Envía un mensaje y/o la ruta de una imagen al servidor.
 *
 * @param {{ message: string, imagePath: string|null }} payload
 * @returns {Promise<{ response: string, imagePath: string|null }>}
 */
async function sendChatMessage(payload) {
  const url = `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.chat}`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: payload.message || "",
      imagePath: payload.imagePath || null,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(
      errorText || `Error del servidor (${response.status})`
    );
  }

  const data = await response.json();
  return normalizeChatResponse(data);
}
