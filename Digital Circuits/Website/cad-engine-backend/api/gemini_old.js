export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
        return res.status(500).json({ error: 'GEMINI_API_KEY environment variable is not configured in Vercel.' });
    }

    try {
        const { userPrompt, engineState } = req.body || {};

        const promptWithContext = `
You are an expert Digital Logic CAD Assistant. Address the user simply as "Hello User". Do not include or mention any student roll numbers or IDs.

CURRENT WORKSPACE CAD CONTEXT:
- Logic Expression: ${engineState?.expression || 'N/A'}
- Minimal SOP: ${engineState?.sop || 'N/A'}
- Generated Verilog:
${engineState?.verilog || 'N/A'}

USER QUESTION:
${userPrompt || 'Explain the current circuit state.'}

Formatting Guidelines:
- Keep the response clean, scannable, and natural.
- Avoid raw LaTeX syntax (e.g., avoid $\\sum m$ or $Y = AB + C$). Use plain text or standard notation (e.g., Σm(1, 3, 5, 6, 7) or Y = AB + C).
- Format headers and code blocks cleanly for display.
        `.trim();

        const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key=${apiKey}`;

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                contents: [
                    {
                        parts: [{ text: promptWithContext }]
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
