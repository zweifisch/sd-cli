# API

```js
fetch('http://localhost:8484/v1/images/generations', {
  method: 'POST',
  body: JSON.stringify({
    prompt: "A surreal landscape with floating islands in a pink sky",
  })
}).then(r => r.json()).then(r => console.log(r.data[0].b64_json))
```

