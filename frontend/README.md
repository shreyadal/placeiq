# frontend/

Next.js app with three pages:

- `/` — enter a name, pick a company and a number of questions
- `/interview` — webcam and mic, shows the question, records the answer, sends the audio
  to the backend, shows the score
- `/results` — mastery per concept across sessions, plus speech and camera feedback

```
npm install
npm run dev        # http://localhost:3000, expects the backend on http://127.0.0.1:8000
```

Set `NEXT_PUBLIC_API_URL` to point at a different backend.

Face tracking is built directly on MediaPipe's face landmarks. The math is explained in
[FACE_METRICS.md](FACE_METRICS.md).
