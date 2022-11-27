var data = {{ data | safe }};
console.log(data);
const ctxHDI = document.getElementById('HDI').getContext('2d');

// const myChart = new Chart(ctx, {
//     type: 'line',
//     data: {
//         datasets: [{
//             label: 'HDI',
//             data: data["Human Development Index"],
//             borderColor: 'rgba(255, 0, 0)'
//         },
//         {
//             label: 'PHDI',
//             data: data["PHDI"],
//             borderColor: 'rgba(0, 255, 0)'
//         }]
//     },
//     options: {
//         responsive: true,
//         scales: {
//             y: {
//                 beginAtZero: false,
//                 position: 'right'
//             }
//         }
//     }
// });
const myChart = new Chart(ctxHDI, {
    type: 'line',
    data: {
        datasets: {
            label: 'HDI',
            data: data["Human Development Index"],
            borderColor: 'rgba(255, 0, 0)'
        }
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: false,
                position: 'right'
            }
        }
    }
});