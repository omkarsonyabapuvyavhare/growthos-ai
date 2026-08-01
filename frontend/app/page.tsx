import Link from "next/link";
import { ArrowRight, Compass, HeartPulse, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const STEPS = [
  {
    title: "Share your goal",
    description: "Tell GrowthOS anything you want to learn  -  in your own words.",
    icon: Compass,
  },
  {
    title: "Check in daily",
    description: "Your mood shapes today’s format, duration, and focus  -  not your goal.",
    icon: HeartPulse,
  },
  {
    title: "Grow with evidence",
    description: "Reflect after each session so the next plan gets smarter.",
    icon: Sparkles,
  },
];

export default function HomePage() {
  return (
    <div className="space-y-16">
      <section className="relative overflow-hidden rounded-3xl border border-blue-100/80 bg-white/70 px-6 py-14 shadow-sm backdrop-blur-sm dark:border-slate-800 dark:bg-slate-900/60 sm:px-12 sm:py-20">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_80%_20%,rgba(59,130,246,0.18),transparent_45%)]" />
        <div className="relative max-w-2xl space-y-6">
          <p className="text-sm font-semibold uppercase tracking-[0.22em] text-blue-600 dark:text-blue-300">
            GrowthOS AI
          </p>
          <h1 className="text-4xl font-semibold tracking-tight text-slate-900 sm:text-5xl dark:text-white">
            Learn anything you want, in a way that fits who you are and how you feel today.
          </h1>
          <p className="max-w-xl text-base leading-relaxed text-slate-600 sm:text-lg dark:text-slate-300">
            GrowthOS optimizes for growth, not attention. A few focused resources.
            Clear reasons for every task. Progress from completion and reflection  - 
            while your long-term goal stays unchanged.
          </p>
          <div className="flex flex-wrap gap-3">
            <Button asChild size="lg">
              <Link href="/onboarding">
                Start My Growth Journey
                <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link href="#how-it-works">See How It Works</Link>
            </Button>
          </div>
        </div>
      </section>

      <section id="how-it-works" className="space-y-6 scroll-mt-24">
        <div className="max-w-2xl">
          <h2 className="text-2xl font-semibold tracking-tight">How it works</h2>
          <p className="mt-2 text-slate-600 dark:text-slate-300">
            One intelligent coach. Three simple steps. No feed. No rabbit holes.
          </p>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {STEPS.map((step, index) => {
            const Icon = step.icon;
            return (
              <Card key={step.title} className="transition-transform hover:-translate-y-0.5">
                <CardHeader>
                  <div className="mb-2 flex h-10 w-10 items-center justify-center rounded-xl bg-blue-50 text-blue-700 dark:bg-slate-800 dark:text-blue-300">
                    <Icon className="h-5 w-5" aria-hidden />
                  </div>
                  <CardTitle className="text-base">
                    <span className="mr-2 text-blue-600 dark:text-blue-300">{index + 1}.</span>
                    {step.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-slate-600 dark:text-slate-300">{step.description}</p>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>
    </div>
  );
}
