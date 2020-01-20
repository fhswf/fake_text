function showAnswers(answers) {
    let answerdiv = document.getElementById('answers');
    console.log(answers)

    while (answerdiv.firstChild) {
        answerdiv.removeChild(answerdiv.firstChild);
    }

    answers.forEach(element => {
        answerdiv.insertAdjacentHTML('afterbegin', "<article><p>" + element + "</p></article>")
    });

    answerdiv.hidden = false;
}

function submit() {
    let story = document.getElementById("story");
    console.log(story.value);

    const url = 'https://jupiter.fh-swf.de/fake_text/text';
    data = { text: story.value + " " };
    fetch(url, {
        method: 'POST',
        mode: 'cors',
        credentials: 'same-origin',
        headers: {
            'Content-Type': 'application/json'
        },
        redirect: 'follow',
        referrerPolicy: 'no-referrer',
        body: JSON.stringify(data)
    })
        .then(response => response.json())
        .then(json => {
            showAnswers(json.result);
        })
        .catch(err => console.log("an error occurred: %o", err))
}