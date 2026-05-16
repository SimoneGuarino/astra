# Frontend UI Implementation Plan

## Architecture Pattern
Follow existing React/TypeScript patterns in the project. Meeting panel uses same design system as DesktopAgentPanel (Tailwind, cards, tabs, grid layout). Meeting hooks use same pattern as useAssistantSession (useState, useEffect, Tauri invoke).

## Component Structure

### File: `src/components/MeetingLivePanel.tsx`
- Main live dashboard panel
- Tab-based layout: Transcript | Summary | Actions | Decisions | Notes
- Call status bar at the top
- Meetin  panel with live transcript, summary, actions, decisions, notes and export controls
- Same design tokens as DesktopAgentPanel (bg-white/60, backdrop-blur-sm, etc.)

### File: `src/hooks/useMeetingEngine.ts`
- Rust Tauri invoke bindings for meeting commands
- Same pattern as useDesktopAgent (useCallback, invoke)
- Meeting state management

### File: `src/hooks/useLiveTranscript.ts`
- Real-time transcript management
- Auto-scroll like AssistantChat
- Speaker highlighting

### File: `src/hooks/useMeetingSummary.ts`
- Summary panel logic
- Live updates during call
- Post-call summary card

### File: `src/components/MeetingSummary.tsx`
- Post-call summary card displayed after meeting ends
- Action items list
- Decision log
- Export controls (JSON, Markdown, CSV)

### File: `src/types/meeting.ts`
- TypeScript types matching Rust types

## Component Hierarchy

```
MeetingLivePanel (if isOpen)
├── CallStatusBar (platform, elapsed time, participants)
├── TabNav (Transcript | Summary | Actions | Decisions | Notes)
├── TranscriptView (live transcript with auto-scroll)
├── SummaryView (rolling summary, post-call summary)
├── ActionItemsView (extracted action items with assignees/deadlines)
├── DecisionsView (logged decisions)
├── NotesView (user notes, export controls)
└── ExportBar (JSON, Markdown, CSV export buttons)
```

## Design Specifications

### Call Status Bar
- Height: 40px
- Background: green/purple/red dot indicator for meeting state
- Text: platform name, elapsed time, participant count

### Tab Navigation
- Background: bg-white/40
- Active tab: white background, border-bottom
- Inactive tabs: gray text

### Transcript View
- Font: code font for transcript
- Speaker name: colored circle + name
- Message text: left-aligned
- Auto-scroll to latest entry

### Summary View
- Card layout with summary text
- Live rolling summary indicator
- Post-call detailed summary section

### Action Items View
- Card per action item
- Assignee, deadline, status (open/closed)
- Check-box to mark complete

### Decisions View
- Card per decision
- Date, rationale, decision maker

### Notes View
- Rich text editor for user notes
- Export buttons (JSON, Markdown, CSV)

## Tauri Commands to invoke (from useMeetingEngine.ts)

```typescript
import { invoke } from '@tauri-apps/api/core';

export interface MeetingSession {
  session_id: string;
  platform: string;
  status: string;
  started_at: number;
}

export const useMeetingEngine = () => {
  const detectActiveCall = useCallback(() => invoke<boolean>('meeting_detect_active_call'), []);
  const startMeeting = useCallback(() => invoke<MeetingSession>('meeting_start'), []);
  const stopMeeting = useCallback(() => invoke<void>('meeting_stop'), []);
  const pauseMeeting = useCallback(() => invoke<void>('meeting_pause'), []);
  const resumeMeeting = useCallback(() => invoke<void>('meeting_resume'), []);
  const getSummary = useCallback(() => invoke<any[]>('meeting_get_summary'), []);
  const getTranscript = useCallback(() => invoke<any[]>('meeting_get_transcript'), []);
  const getActionItems = useCallback(() => invoke<any[]>('meeting_get_action_items'), []);
  const getDecisions = useCallback(() => invoke<any[]>('meeting_get_decisions'), []);
  const exportJson = useCallback(() => invoke<string>('meeting_export_json'), []);
  const exportMarkdown = useCallback(() => invoke<string>('meeting_export_markdown'), []);
  const getStatus = useCallback(() => invoke<string>('meeting_get_status'), []);
  return { detectActiveCall, startMeeting, stopMeeting, pauseMeeting, resumeMeeting, getSummary, getTranscript, getActionItems, getDecisions, exportJson, exportMarkdown, getStatus };
};
```

## Integration into App.tsx

Add meeting button to toolbar/header:
- Icon: Mic/Camera icon
- Position: right side of header, next to existing controls
- On click: opens MeetingLivePanel

## File Locks

- `src/components/MeetingLivePanel.tsx` — Frontend UI Agent
- `src/components/MeetingSummary.tsx` — Frontend UI Agent
- `src/hooks/useMeetingEngine.ts` — Frontend UI Agent
- `src/hooks/useLiveTranscript.ts` — Frontend UI Agent
- `src/hooks/useMeetingSummary.ts` — Frontend UI Agent
- `src/types/meeting.ts` — Frontend UI Agent
- `App.tsx` — Frontend UI Agent (modify)

## Validation Checklist

- [ ] All TypeScript types compile correctly
- [ ] MeetingLivePanel mounts without errors
- [ ] Tabs render as expected
- [ ] Live transcript displays correctly
- [ ] Summary updates correctly
- [ ] Action items display correctly
- [ ] Decisions display correctly
- [ ] Export controls work
- [ ] No CSS conflicts with existing components
- [ ] No regressions in existing features
