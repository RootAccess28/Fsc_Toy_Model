export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
        return res.status(500).json({ error: 'GEMINI_API_KEY environment variable is not configured in Vercel.' });
    }

    try {
        // 1. Extract the imageBase64 and mimeType along with existing data
        const { userPrompt, engineState, imageBase64, mimeType } = req.body || {};

        const promptWithContext = `
You are an expert Digital Logic CAD Assistant for the User.
Answer in a human readable text.

CURRENT WORKSPACE CAD CONTEXT:
- Logic Expression: ${engineState?.expression || 'N/A'}
- Minimal SOP: ${engineState?.sop || 'N/A'}
- Generated Verilog:
${engineState?.verilog || 'N/A'}

USER QUESTION:
${userPrompt || 'Explain the current circuit state.'}

Keep your explanation clear, technically accurate, and concise.
        `.trim();

        // 2. Build the message parts dynamically
        const parts = [
            { text: promptWithContext }
        ];

        // 3. If an image is uploaded, append it to the payload
        if (imageBase64 && mimeType) {
            parts.push({
                inlineData: {
                    mimeType: mimeType,
                    data: imageBase64
                }
            });
        }

        const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key=${apiKey}`;

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                contents: [
                    {
                        parts: parts // Send both text and image parts
                    }
                ]
            })
        });

        const data = await response.json();

        if (data.error) {
            console.error("Gemini API Error:", data.error);
            return res.status(500).json({ error: data.error.message || 'Error from Google Gemini API.' });
        }

        const reply = data.candidates?.[0]?.content?.parts?.[0]?.text;

        if (!reply) {
            return res.status(500).json({ error: 'No text returned from Gemini API.' });
        }

        return res.status(200).json({ reply });
    } catch (err) {
        console.error("Handler Exception:", err);
        return res.status(500).json({ error: err.message || 'Internal Server Error' });
    }
}
