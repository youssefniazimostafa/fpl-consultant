import Markdown from "react-markdown";

// Renders a Live consultant answer. react-markdown builds React elements
// (never innerHTML) and, without rehype-raw, raw HTML in the text is not
// parsed as HTML. skipHtml drops HTML nodes; only the simple elements
// below are allowed -- links, images, headings, code and tables are
// unwrapped to their plain text, so model output can never run as code.

const ALLOWED = ["p", "strong", "em", "ul", "ol", "li", "br"];

// "<b>x</b>" becomes the literal text "&lt;b&gt;x&lt;/b&gt;" for markdown,
// so a model answer that contains HTML is shown as text, not dropped.
const escapeHtml = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export function ChatMarkdown({ text }: { text: string }) {
  return (
    <div className="chat-md">
      <Markdown allowedElements={ALLOWED} unwrapDisallowed skipHtml>
        {escapeHtml(text)}
      </Markdown>
    </div>
  );
}
