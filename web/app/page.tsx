"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import * as Tooltip from "@radix-ui/react-tooltip";
import { useState } from "react";
import { Shell } from "@/components/Shell";

export default function Page() {
  const [qc] = useState(() => new QueryClient());
  return (
    <QueryClientProvider client={qc}>
      <Tooltip.Provider>
        <Shell />
      </Tooltip.Provider>
    </QueryClientProvider>
  );
}
