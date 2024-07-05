# Change out TK for Gradios

Dunno why I didn't just use Gradios from the start...

# Using other bots

Even though I'm saying you can use other bots, you will have to modify the result extraction code atm.

I've really only made the url and json bodys parameters... 

# You're talking about memory

Ollama has a nice mechanism to support history, but I need to come up with a good way to handle the template.

```
 "messages": [
   {
     "role": "user",
     "content": "why is the sky blue?"
   },
   {
     "role": "assistant",
     "content": "due to rayleigh scattering."
   },
   {
     "role": "user",
     "content": "how is that different than mie scattering?"
   }
 ]
```

This is particullarly a challenge when it comes to incorporating a different LLM.
