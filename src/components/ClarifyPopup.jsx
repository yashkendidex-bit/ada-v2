import React, { useEffect, useRef, useState } from 'react';

/**
 * ADA's clarifying question — Level 3's clarifier, on screen.
 *
 * Deliberately not a yes/no gate: a build that is under-specified needs an *answer*,
 * not permission. Options are clickable because most real answers are one of three
 * sizes, and the text field is there because the useful answer is often "it bolts to
 * a 40 mm shaft", which no set of chips predicted. Both routes emit the same event,
 * and the voice path — just saying the answer out loud — resolves the same future, so
 * a build never hangs behind a box nobody is looking at.
 */
const ClarifyPopup = ({ request, onAnswer }) => {
    const [text, setText] = useState('');
    const inputRef = useRef(null);

    useEffect(() => {
        setText('');
        if (request) {
            const t = setTimeout(() => inputRef.current?.focus(), 200);
            return () => clearTimeout(t);
        }
    }, [request]);

    if (!request) return null;

    const answer = (value) => onAnswer((value ?? text).trim() || request.default || '');

    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in">
            <div className="relative w-full max-w-lg p-8 bg-black/90 border border-cyan-500/30 rounded-3xl shadow-[0_0_50px_rgba(34,211,238,0.15)] backdrop-blur-2xl">
                <div className="flex items-center gap-4 mb-6 relative z-10">
                    <div className="p-3 rounded-full bg-cyan-900/30 border border-cyan-500/50 text-cyan-400 shadow-[0_0_15px_rgba(34,211,238,0.2)]">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                    </div>
                    <div>
                        <h2 className="text-xl font-bold text-cyan-400 tracking-wider font-mono drop-shadow-sm">
                            ONE THING I NEED
                        </h2>
                        <p className="text-xs text-cyan-600 font-mono tracking-widest uppercase">
                            Before I start building
                        </p>
                    </div>
                </div>

                <div className="mb-6 space-y-4 relative z-10">
                    <p className="text-gray-200 leading-relaxed text-base">{request.question}</p>
                    {request.why && (
                        <p className="text-xs text-cyan-600 font-mono tracking-wide">
                            asked because {request.why}
                        </p>
                    )}

                    {request.choices?.length > 0 && (
                        <div className="flex flex-wrap gap-2">
                            {request.choices.map((choice) => (
                                <button
                                    key={choice}
                                    onClick={() => answer(choice)}
                                    className="px-3 py-2 rounded-lg border border-cyan-800/60 bg-cyan-950/40 text-cyan-300 text-xs font-mono hover:bg-cyan-900/60 hover:border-cyan-500 hover:text-cyan-200 transition-all"
                                >
                                    {choice}
                                </button>
                            ))}
                        </div>
                    )}

                    <div className="flex gap-2">
                        <input
                            ref={inputRef}
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter') answer();
                                if (e.key === 'Escape') answer(request.default);
                            }}
                            placeholder={request.default ? `or type it — default: ${request.default}` : 'or type it'}
                            className="flex-1 px-4 py-3 rounded-xl bg-cyan-950/30 border border-cyan-800/50 text-white text-sm font-mono placeholder-gray-600 focus:outline-none focus:border-cyan-500/70"
                        />
                        <button
                            onClick={() => answer()}
                            className="px-5 py-3 rounded-xl border border-cyan-500/30 bg-cyan-950/40 text-cyan-400 hover:bg-cyan-900/60 hover:border-cyan-400 hover:text-cyan-300 transition-all duration-200 font-bold tracking-wider uppercase text-xs"
                        >
                            Submit
                        </button>
                    </div>
                </div>

                <div className="flex justify-end relative z-10">
                    <button
                        onClick={() => answer(request.default)}
                        className="px-4 py-2 rounded-lg border border-gray-700/60 bg-gray-900/40 text-gray-400 hover:text-gray-200 hover:border-gray-500 transition-all font-mono text-xs uppercase tracking-wider"
                    >
                        Use the default and build
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ClarifyPopup;
