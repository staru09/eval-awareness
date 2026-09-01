import { Fragment, type ReactNode } from "react";

// Lightweight inline-markdown renderer for transcript text: **bold**,
// *italic*/_italic_, `code`, and paragraph/line breaks. Model output in
// these transcripts routinely contains literal markdown (numbered lists
// with **bold** labels, etc.) that should render, not show as raw asterisks.
const INLINE_RE = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|_[^_]+_)/g;

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const parts = text.split(INLINE_RE);
  return parts.map((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={key}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={key}>{part.slice(1, -1)}</code>;
    }
    if (
      (part.startsWith("*") && part.endsWith("*")) ||
      (part.startsWith("_") && part.endsWith("_"))
    ) {
      return <em key={key}>{part.slice(1, -1)}</em>;
    }
    return <Fragment key={key}>{part}</Fragment>;
  });
}

export function FormattedText({ text }: { text: string }) {
  const paragraphs = text.split(/\n{2,}/);
  return (
    <>
      {paragraphs.map((para, pi) => (
        <p key={pi}>
          {para.split("\n").map((line, li, arr) => (
            <Fragment key={li}>
              {renderInline(line, `${pi}-${li}`)}
              {li < arr.length - 1 && <br />}
            </Fragment>
          ))}
        </p>
      ))}
    </>
  );
}
