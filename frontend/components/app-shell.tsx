"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { LayoutDashboard, Map, Moon, Sparkles, Sun, Target } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { getGoalTitle, getProgressPercent, getUserId } from "@/lib/user-state";
import { cn } from "@/lib/utils";
import { useTheme } from "next-themes";

const NAV = [
  { href: "/roadmap", label: "Roadmap", icon: Map },
  { href: "/today", label: "Today", icon: Target },
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [goalTitle, setGoalTitle] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [hasUser, setHasUser] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    setHasUser(getUserId() !== null);
    setGoalTitle(getGoalTitle());
    setProgress(getProgressPercent());
  }, [pathname]);

  const isLanding = pathname === "/";
  const dark = (resolvedTheme ?? theme) === "dark";

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#dbeafe_0%,_#f8fafc_45%,_#ffffff_100%)] text-slate-900 dark:bg-[radial-gradient(circle_at_top,_#0f172a_0%,_#020617_55%,_#000_100%)] dark:text-slate-50">
      <header className="sticky top-0 z-40 border-b border-blue-100/70 bg-white/70 backdrop-blur-md dark:border-slate-800 dark:bg-slate-950/70">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-600 text-white shadow-sm">
              <Sparkles className="h-4 w-4" aria-hidden />
            </span>
            <span className="text-lg">GrowthOS AI</span>
          </Link>

          {!isLanding && hasUser ? (
            <nav className="hidden items-center gap-1 md:flex" aria-label="Primary">
              {NAV.map((item) => {
                const Icon = item.icon;
                const active = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium transition-colors",
                      active
                        ? "bg-blue-600 text-white"
                        : "text-slate-600 hover:bg-blue-50 dark:text-slate-300 dark:hover:bg-slate-800",
                    )}
                  >
                    <Icon className="h-4 w-4" aria-hidden />
                    {item.label}
                  </Link>
                );
              })}
            </nav>
          ) : null}

          <div className="flex items-center gap-2">
            {mounted ? (
              <Button
                type="button"
                variant="ghost"
                size="icon"
                aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
                onClick={() => setTheme(dark ? "light" : "dark")}
              >
                {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </Button>
            ) : (
              <div className="h-10 w-10" />
            )}
          </div>
        </div>

        {!isLanding && hasUser ? (
          <div className="mx-auto max-w-6xl space-y-2 px-4 pb-3 sm:px-6">
            {goalTitle ? (
              <p className="truncate text-sm text-slate-600 dark:text-slate-300">
                <span className="font-medium text-slate-800 dark:text-slate-100">Goal:</span>{" "}
                {goalTitle}
              </p>
            ) : null}
            <div className="flex items-center gap-3">
              <Progress value={progress} className="h-2" aria-label="Roadmap progress" />
              <span className="shrink-0 text-xs font-medium text-slate-500 dark:text-slate-400">
                {Math.round(progress)}%
              </span>
            </div>
            <nav className="flex gap-2 overflow-x-auto md:hidden" aria-label="Mobile">
              {NAV.map((item) => {
                const active = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "rounded-full px-3 py-1.5 text-xs font-medium",
                      active
                        ? "bg-blue-600 text-white"
                        : "bg-white/80 text-slate-600 dark:bg-slate-900 dark:text-slate-300",
                    )}
                  >
                    {item.label}
                  </Link>
                );
              })}
            </nav>
          </div>
        ) : null}
      </header>

      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6 sm:py-10">{children}</main>
    </div>
  );
}
