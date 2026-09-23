// Standalone SPA entry (no Next.js runtime): used to publish the replay dashboard as a single static page.
import { createRoot } from "react-dom/client";
import Page from "@/app/page";

createRoot(document.getElementById("root")!).render(<Page />);
