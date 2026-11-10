import type { Metadata } from 'next';
import './globals.css';

// Fonts load via Google Fonts <link> (next/font's build-time fetch is
// blocked in this sandbox). If the fonts ever fail to load at runtime,
// the stacks in globals.css fall back to system fonts — the layout never
// breaks.
export const metadata: Metadata = {
  title: 'Productivity Monitor — AI Employee Monitoring',
  description:
    'Browser-only employee productivity monitor upgraded from the open-source AI-powered employee productivity monitoring system (MIT). Working and away-from-desk detection, live from your webcam.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700;900&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
